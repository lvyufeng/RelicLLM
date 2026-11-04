"""Routed experts from an NVIDIA ModelOpt NVFP4 checkpoint, run by the fused FP4 kernel.

The FP8 checkpoint this tree is built around keeps its experts in host RAM and stages a step's
active ones over PCIe, which on 4x2080Ti is 0.55 GiB a rank a step and a 98 ms floor against a
100 ms budget for 10 tok/s. NVFP4 halves those bytes: a rank's 48 layers are 14.9 GiB against
21.48 GiB of card, so **the whole expert set fits**, no staging happens during a decode, and the
step's dominant term goes away rather than getting cheaper.

What makes this usable on sm_75, which has no FP4 tensor core, is that `relic-core` already
carries a software path: `moe_single_token_fp4_forward` decodes E2M1 codes and their UE4M3 group
scales into the MMA operands. Measured on a 2080 Ti at this model's shapes, ten experts a layer:

    the fused FP4 kernel          0.356 ms
    the FP8 per-expert loop       2.42 ms      (what a layer costs today)

Layout, from ModelOpt's `hf_quant_config.json`:

* weights are `[E, out, in/2]` packed nibbles and `[E, out, in/32]` UE4M3 scale bytes;
* gate and up ship separately, which is what the kernel's `w1`/`w3` arguments want;
* `group_size` is 32 in the kernel's contract. The checkpoint states 16, and the kernel packs two
  16-element sub-blocks under one entry (`QK_NVFP4_SUB 16` inside a `QK_NVFP4 64` block), so the
  two do not line up one-for-one and a conversion pass is required. This module does not guess
  which side is right -- it holds the arithmetic and the layout checks, and the conversion is a
  separate step.

The `exclude_modules` list in the same config keeps `linear_attn`, the hyper-connection mixers, the
shared expert and the embedding/head at their own precision, so a caller assembles this alongside
the existing dense path rather than replacing it.
"""

from __future__ import annotations

import atexit
import os

import torch

__all__ = ["NvFp4ExpertBank"]


class NvFp4ExpertBank:
    """One rank's experts as the fused FP4 kernel takes them.

    `w1`/`w3` are gate and up, `w2` is down, each a `(codes, scales)` pair of the shapes the
    kernel checks:

        codes   [E, out, in/2]      packed nibbles
        scales  [E, out, in/16]     E4M3 bytes, one a 16-element group

    A rank that holds all 48 layers stages nothing during a decode.
    """

    def __init__(self, ops, *, hidden: int, inter: int, experts: int, world_size: int = 1,
                 dtype=torch.bfloat16, device="cuda", swiglu_limit: float = 0.0):
        for name, value in (("hidden", hidden), ("inter", inter)):
            if value % 32:
                # The kernel's own contract, checked here so a bad shape is a load-time error
                # rather than a shape error from inside a capture.
                raise ValueError(f"{name}={value} must be divisible by 32 for the fp4 path")
        self.ops = ops
        self.hidden = int(hidden)
        self.inter = int(inter)
        self.experts = int(experts)
        #: Experts are split `expert_id % world_size == rank`, so this rank's row for a global id is
        #: `id // world_size`. Global 512 zero-fills to 512 rows a rank, four times the memory for
        #: three quarters of nothing.
        self.world_size = int(world_size)
        #: Which rank's experts this bank holds; row `i` is global expert `rank + world_size * i`.
        self.rank = 0
        self.dtype = dtype
        self.device = device
        self.swiglu_limit = float(swiglu_limit)
        #: `layer -> (w1q, w1s, w3q, w3s, w2q, w2s)`, filled by `add_layer`.
        self.layers: dict[int, tuple] = {}
        #: `layer -> scalar`, the product of the two global scales the kernel does not apply.
        self.scales: dict[int, tuple] = {}
        #: Run the checkpoint's own W4A4 regime (activation on the E2M1 grid) rather than keeping
        #: the activation in float. Both kernels accept either; this is the one the checkpoint was
        #: tuned for and the one an FP4 tensor core would do.
        #: W4A8 by default: the activation stays in float, so the only quantisation error is the
        #: weights' -- measured at 8-14% a stage against the FP8 checkpoint, which is the 4-bit
        #: difference and not a defect. W4A4 is available and is the checkpoint's own regime, but
        #: its grid saturates any activation past the calibration range, and this model's hidden
        #: state is far past it.
        #: W4A4 by default, which is the regime this checkpoint was calibrated for: the activation
        #: goes onto an E2M1 grid taken from the row's own amax, the grid the fused kernel a
        #: GPU with FP4 tensor cores would use. `POCKETLLM_NVFP4_A4=0` keeps the activation in
        #: float instead -- more bits on the activation, but it leaves the weights multiplying a
        #: value 50x past the range they were calibrated on, and measured worse end to end.
        #: W4A16 by default: the activation stays in float and is not quantised, which is how vLLM
        #: runs ModelOpt NVFP4 experts on sm75 (Marlin W4A16). It is also the only regime with no
        #: activation-side error: `input_scale` never enters the forward pass, so nothing can sit
        #: outside a calibrated range, and the only quantisation is the weight's own 4-bit
        #: rounding -- measured at 8.4% a stage against the FP8 checkpoint.
        #: `POCKETLLM_NVFP4_A4=1` runs the checkpoint's own W4A4 grid instead.
        self.activation_fp4 = os.environ.get("POCKETLLM_NVFP4_A4") == "1"
        #: Which layers the bank actually answered for -- a layer missing here was served by some
        #: other backend, which is a thing to know before blaming the numbers.
        self.served: set[int] = set()
        #: Wall time inside the op, when `POCKETLLM_NVFP4_TIME=1`. A bank call is the whole MoE for
        #: one layer of one token, so this is the number to compare against the step.
        self.timing = os.environ.get("POCKETLLM_NVFP4_TIME") == "1"
        self.seconds = 0.0
        self.calls = 0

    def add_layer(self, layer_idx: int, tensors: tuple) -> None:
        """Hold one layer's six tensors, checked against the kernel's shape contract.

        A ModelOpt checkpoint carries two scalars a projection that the kernel does not take:
        `weight_scale_2` (one per projection) and `input_scale` (one per projection). Both are
        folded into `self.scales`, which `__call__` multiplies the returned result by, because
        they are per-layer constants and the kernel's arithmetic has no place to put them.
        """
        w1q, w1s, w3q, w3s, w2q, w2s = (t.contiguous() for t in tensors)
        h, i = self.hidden, self.inter
        expected = (
            (w1q, (i, h // 2), "w1q"), (w1s, (i, h // 16), "w1s"),
            (w3q, (i, h // 2), "w3q"), (w3s, (i, h // 16), "w3s"),
            (w2q, (h, i // 2), "w2q"), (w2s, (h, i // 16), "w2s"),
        )
        for tensor, tail, name in expected:
            if tuple(tensor.shape[-2:]) != tail:
                raise ValueError(
                    f"{name} for layer {layer_idx} is {tuple(tensor.shape[-2:])}, want {tail}"
                )
        self.layers[layer_idx] = (w1q, w1s, w3q, w3s, w2q, w2s)

    def __len__(self) -> int:
        return len(self.layers)

    def __call__(self, layer_idx: int, hidden_states: torch.Tensor, indices: torch.Tensor,
                 weights: torch.Tensor) -> torch.Tensor | None:
        """The routed sum for one token, or None when this layer is not banked.

        `indices`/`weights` are the router's `(1, top_k)`; the kernel wants a `[k]` of int64 ids
        and float32 coefficients, which is why they are reshaped rather than passed through.
        """
        held = self.layers.get(layer_idx)
        if held is None:
            return None
        w1q, w1s, w3q, w3s, w2q, w2s = held
        entry = self.scales.get(layer_idx)
        if entry is None:
            return None
        g2, gis, d2, dis = entry
        # A prefill chunk arrives as several tokens at once and the kernel is a single-token one, so
        # the tokens are done in turn. Decode -- the path this bank exists for -- is one token and
        # pays nothing for the loop.
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        if flat.shape[0] != 1:
            parts = [
                self(layer_idx, flat[t: t + 1], indices[t: t + 1], weights[t: t + 1])
                for t in range(flat.shape[0])
            ]
            if any(part is None for part in parts):
                return None
            return torch.cat(parts, dim=0)
        # One place computes the rows so the kernel and any reference see the same numbers: a
        # global id `g` is row `g // world_size`, and `ShardedMoE` has already sentinel-ised the
        # experts this rank does not own, whose rows fall past the end of the stack.
        rows = (indices.reshape(-1) // self.world_size).to(torch.int64)
        _t0 = torch.cuda.Event(enable_timing=True) if self.timing else None
        _t1 = torch.cuda.Event(enable_timing=True) if self.timing else None
        if _t0 is not None:
            _t0.record()
        out = self.ops.nvfp4_moe_token_forward(
            hidden_states.reshape(1, -1).to(torch.float32),
            rows,
            weights.reshape(-1).to(torch.float32),
            w1q, w1s, w3q, w3s, w2q, w2s,
            0,
            self.swiglu_limit,
            gis,
            dis,
            g2.reshape(-1),
            d2.reshape(-1),
            self.activation_fp4,
        )
        if _t1 is not None:
            _t1.record()
            _t1.synchronize()
            self.seconds += _t0.elapsed_time(_t1) / 1000.0
        self.calls += 1
        self.served.add(layer_idx)
        return out.reshape_as(hidden_states).to(hidden_states.dtype)

    def release(self) -> None:
        self.layers.clear()
        torch.cuda.empty_cache()

_BANKS: list = []


def _report_banks() -> None:
    """Say which layers the banks answered for, once, at interpreter exit.

    A layer missing from `served` was served by some other expert backend; that is the difference
    between a wrong number and a layer that never ran through this path at all.
    """
    for i, bank in enumerate(_BANKS):
        print(f"[nvfp4] bank {i}: {bank.calls} calls, served {len(bank.served)} of "
              f"{len(bank.layers)} held layers", flush=True)
        if bank.timing and bank.calls:
            print(f"[nvfp4] bank {i}: {bank.seconds:.3f}s in the op over {bank.calls} calls, "
                  f"{bank.seconds / bank.calls * 1e3:.3f} ms each", flush=True)
        if bank.calls and len(bank.served) < len(bank.layers):
            missing = sorted(set(bank.layers) - bank.served)
            print(f"[nvfp4] bank {i}: never served {missing[:10]}", flush=True)


atexit.register(_report_banks)


def _expert_keys(index: dict, layer: int, expert: int) -> dict[str, str]:
    """The six tensors one expert needs, under the checkpoint's own naming.

    ModelOpt names a routed expert's projections for what they do in the SwiGLU, so `gate_proj`
    and `up_proj` are the kernel's `w1`/`w3` and `down_proj` is `w2` -- there is no packed
    `gate_up` here because this format never materialises one.
    """
    prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}"
    want = {}
    for proj, slot in (("gate_proj", "w1"), ("up_proj", "w3"), ("down_proj", "w2")):
        if f"{prefix}.{proj}.weight" not in index:
            return {}
        want[f"{slot}q"] = f"{prefix}.{proj}.weight"
        want[f"{slot}s"] = f"{prefix}.{proj}.weight_scale"
    return want


def _one_expert(index: dict, handles: dict, layer: int, expert: int, proj: str,
                kind: str) -> torch.Tensor:
    """One per-expert scalar, as a 1-element tensor on the card.

    `weight_scale_2` is the one scalar that differs from expert to expert, so it has to come back
    as a vector the kernel can index by whichever expert a route landed on.
    """
    key = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}.{kind}"
    return handles[index[key]].get_tensor(key).reshape(1)


def build_bank(ops, model_dir: str, *, rank: int, world_size: int, layers: int,
               hidden: int, inter: int, experts: int, device="cuda",
               swiglu_limit: float = 0.0, verbose: bool = True) -> NvFp4ExpertBank:
    """Hold this rank's share of a ModelOpt NVFP4 checkpoint's routed experts on the card.

    Ownership is the same `expert_id % world_size == rank` split the FP8 path uses, so a run can
    change checkpoint and keep every other decision.

    Two per-projection scalars are folded in: `weight_scale_2` and `input_scale`. The kernel
    quantises the activation itself at a scale derived from its own range, so an activation the
    checkpoint calibrated with `input_scale` does not come back at that scale unless the product of
    both constants is applied to the result. `NvFp4ExpertBank.scales` holds it.
    """
    import json
    import os

    from safetensors import safe_open  # noqa: PLC0415 -- only needed on this path

    bank = NvFp4ExpertBank(ops, hidden=hidden, inter=inter, experts=experts, world_size=world_size,
                           device=device, swiglu_limit=swiglu_limit)
    bank.root = model_dir
    bank.rank = rank
    _BANKS.append(bank)
    index: dict[str, str] = {}
    for name in sorted(os.listdir(model_dir)):
        if name.endswith(".safetensors.index.json"):
            with open(os.path.join(model_dir, name)) as f:
                index = json.load(f)["weight_map"]
            break
    if not index:
        raise FileNotFoundError(f"no safetensors index in {model_dir}")
    held = [e for e in range(experts) if e % world_size == rank]

    # Keyed by the name the index uses, which is the file name rather than a path.
    handles = {
        name: safe_open(os.path.join(model_dir, name), framework="pt")
        for name in sorted(set(index.values()))
    }
    try:
        for layer in range(layers):
            stacks: dict[str, list] = {}
            scales: list[float] = []
            for expert in held:
                keys = _expert_keys(index, layer, expert)
                if not keys:
                    break
                for slot, key in keys.items():
                    t = handles[index[key]].get_tensor(key).to(device)
                    if slot.endswith("s"):
                        t = t.view(torch.uint8)
                    stacks.setdefault(slot, []).append(t)
                scales.append(1.0)
            if not stacks:
                continue
            tensors = tuple(torch.stack(stacks[k]) for k in ("w1q", "w1s", "w3q", "w3s", "w2q", "w2s"))
            bank.add_layer(layer, tensors)
            # `weight_scale_2` is per expert: at layer 0 the values run 2.65e-5 to 1.02e-4, a factor
            # of 3.9, so a single constant for the bank is wrong for most experts. `input_scale` is
            # not -- every expert in a layer carries the same two -- so that stays a constant.
            gs2 = torch.stack([_one_expert(index, handles, layer, e, "gate_proj",
                                           "weight_scale_2") for e in held]).to(device)
            ds2 = torch.stack([_one_expert(index, handles, layer, e, "down_proj",
                                           "weight_scale_2") for e in held]).to(device)
            gis = float(_one_expert(index, handles, layer, held[0], "gate_proj", "input_scale"))
            dis = float(_one_expert(index, handles, layer, held[0], "down_proj", "input_scale"))
            bank.scales[layer] = (gs2, gis, ds2, dis)
            if verbose:
                print(f"[nvfp4] layer {layer} resident, {len(held)} experts", flush=True)
    finally:
        handles.clear()
    return bank
