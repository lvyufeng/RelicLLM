"""The Engram runs entirely on the host when the block around it runs on an Ascend card.

The direct user requirement is that the Engram -- not only its 91.55 GiB table, but the projection,
the gate and every reduce over them -- executes on the host for Ascend inference. Two small changes
implement that, and this file is what pins them:

* `Block.__init__` builds the Engram through `modules._engram_device`, which answers `cpu` for an
  `npu` block and passes every other device through unchanged;
* `Engram.forward` carries the current token span to the host, runs the unchanged tiled arithmetic
  there (`_forward_span`), and carries the result back. When the Engram is already where the
  activation is it does nothing extra at all.

What this file does **not** claim:

* no device work happens here. Most of this file is structural: a stand-in device object proves the
  *choice* without needing `torch_npu`, and a shim tensor proves the *boundary* without needing an
  accelerator. The end-to-end case runs only when a leased card is named through
  `RELIC_ASCEND_ENGRAM_DEVICE`, and that one is the lead's to run;
* the CPU and CUDA paths are unchanged, and the assertions are bit-equality against the arithmetic
  those paths ran before the offload -- not a tolerance;
* nothing here relaxes a capability guard, changes a quantization format or edits the table store.
  The host Engram runs the same `quantized_linear` a CPU run does, including whatever source-code
  metadata the loader hung on the host parameters.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from relicllm.models.deepseek_v4_1 import modules as modules_module
from relicllm.models.deepseek_v4_1.modules import Engram, ResidentEngramTable

DIM = 32
HC_MULT = 2
HEAD_DIM = 32
ROWS = 64
N_HEADS = 2
MAX_NGRAM = 2


class _Cfg(SimpleNamespace):
    """Exactly the fields these constructors read; `Engram` never sees a whole `V41TextConfig`."""


def _cfg() -> _Cfg:
    # `hc_sinkhorn_iters`/`hc_eps` are read by `Block` even when its sublayers are recorders, and
    # `n_layers`/`dtype` are read by `_moe_shape`/the quantized flag on the way to them.
    return _Cfg(dim=DIM, hc_mult=HC_MULT, norm_eps=1e-6, hc_sinkhorn_iters=20, hc_eps=1e-6,
                n_layers=1, dtype="bf16", n_routed_experts=2, n_activated_experts=1)


def _layout() -> SimpleNamespace:
    """The layout attributes `Engram.__init__` reads."""
    return SimpleNamespace(layer_ids=[0], max_ngram_size=MAX_NGRAM, n_heads=N_HEADS, head_dim=HEAD_DIM)


def _hash_columns() -> int:
    return (MAX_NGRAM - 1) * N_HEADS


def _engram(seed: int = 0) -> Engram:
    """A small host Engram with deterministic parameters.

    `device` is left at its default -- the Engram's own host placement, not a stream choice -- so
    these cases exercise exactly the construction the CPU path had and the boundary the Ascend path
    adds.
    """
    torch.manual_seed(seed)
    table = ResidentEngramTable(
        weight=torch.randn(ROWS, HEAD_DIM).to(torch.float8_e4m3fn),
        scale=torch.ones(ROWS, 1).to(torch.float8_e8m0fnu),
    )
    engram = Engram(_cfg(), 0, _layout(), table, quantized=False)
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for parameter in engram.parameters():
            values = torch.randn(parameter.shape, generator=generator, dtype=torch.float32)
            parameter.copy_(values.to(parameter.dtype) * 0.05)
    return engram


def _tokens(engram: Engram, batch: int = 1, tokens: int = 4, seed: int = 3):
    del engram  # only the shapes are read here
    torch.manual_seed(seed)
    x = torch.randn(batch, tokens, HC_MULT, DIM, dtype=torch.bfloat16)
    ids = torch.randint(0, ROWS, (batch, tokens, _hash_columns()))
    mask = torch.ones(batch, tokens, dtype=torch.bool)
    return x, ids, mask


class _FakeAccelerator:
    """A stand-in for `torch.device('npu', 0)` on a host whose torch has no NPU backend.

    It carries only the attributes the placement decision and the recorders below read, which is the
    whole of what they consume. No tensor is ever created on it.
    """

    type = "npu"
    index = 0

    def __repr__(self) -> str:  # pragma: no cover - only shown on a failure
        return "npu:0"

    def __str__(self) -> str:  # pragma: no cover - only shown on a failure
        return "npu:0"


_FAKE = _FakeAccelerator()


# ------------------------------------------------------------------------------------------------
# Where the Engram is built
# ------------------------------------------------------------------------------------------------


def test_the_placement_choice_is_cpu_for_an_ascend_block_and_passthrough_otherwise() -> None:
    """The one decision `Block` consults, made without needing a registered npu backend."""
    assert modules_module._engram_device(_FAKE) == torch.device("cpu")
    # The default and every other real device keep the Engram where the block is; `meta` is a device
    # type that constructs without an accelerator, so it stands in for "some other device" honestly.
    other = torch.device("meta")
    assert modules_module._engram_device(other) is other
    assert modules_module._engram_device(None) is None


def test_a_block_puts_only_its_engram_on_the_host(monkeypatch) -> None:
    """`Block` wiring: an `npu` block hands `cpu` to `Engram`, its own device to everything else.

    Structural rather than numerical -- the sublayer constructors are recorders, and the six raw
    `torch.empty` calls the block makes for its Hyper-Connections coefficients are routed to the host
    because a stand-in device object cannot allocate. What is asserted is exactly the placement
    choice: Engram on the host, attention/FFN/norms/coefficients still on the accelerator.
    """
    seen: dict[str, object] = {}

    def record(name):
        def build(*args, **kwargs):
            seen[name] = kwargs.get("device", "absent")
            return SimpleNamespace(name=name)

        return build

    real_empty = torch.empty

    def empty(*args, **kwargs):
        if kwargs.get("device") is _FAKE:
            return real_empty(*args, **{**kwargs, "device": None})
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(modules_module, "canonical_device", lambda device: _FAKE)
    monkeypatch.setattr(modules_module.torch, "empty", empty)
    for name in ("Attention", "MoE", "Engram", "RMSNorm"):
        monkeypatch.setattr(modules_module, name, record(name.lower()))

    modules_module.Block(_cfg(), 0, 1, 16, engram_table=object(), layout=_layout())

    assert seen["engram"] == torch.device("cpu")
    assert seen["engram"] != _FAKE
    for name in ("attention", "moe", "rmsnorm"):
        assert seen[name] is _FAKE, f"{name} must stay on the block's device"


def test_a_cpu_block_still_builds_its_engram_in_place(monkeypatch) -> None:
    """The default block is unchanged: no device is substituted when the block is on the host."""
    seen: dict[str, object] = {}

    def record(name):
        def build(*args, **kwargs):
            seen[name] = kwargs.get("device", "absent")
            return SimpleNamespace(name=name)

        return build

    for name in ("Attention", "MoE", "Engram", "RMSNorm"):
        monkeypatch.setattr(modules_module, name, record(name.lower()))

    modules_module.Block(_cfg(), 0, 1, 16, engram_table=object(), layout=_layout())

    # `device=None` reaches the block and a host block passes it straight through: the CPU path does
    # not acquire a device it did not have.
    assert seen["engram"] is None


# ------------------------------------------------------------------------------------------------
# The CPU path is unchanged
# ------------------------------------------------------------------------------------------------


def test_a_cpu_engram_calls_the_unchanged_pass_directly(monkeypatch) -> None:
    """No boundary is crossed when the Engram and the activation are both on the host.

    With the tiling forced off, `forward` must be the single pass itself -- same bits, same device,
    same dtype, mask included. The projection is also asserted to receive host operands, which is
    the same statement one level down.
    """
    engram = _engram()
    x, ids, mask = _tokens(engram)
    assert engram.wkv.weight.device.type == "cpu"

    calls: list[tuple[str, str]] = []
    real_linear = modules_module.quantized_linear

    def spy(x_arg, weight, **kwargs):
        calls.append((x_arg.device.type, weight.device.type))
        return real_linear(x_arg, weight, **kwargs)

    monkeypatch.setattr(modules_module, "quantized_linear", spy)
    monkeypatch.setattr(modules_module, "HC_TOKEN_TILE", 0)

    for use_mask in (False, True):
        expected = engram._forward_pass(x, ids, mask if use_mask else None)
        got = engram(x, ids, mask if use_mask else None)
        assert got.device == x.device and got.dtype == x.dtype
        assert torch.equal(got, expected), f"mask={use_mask}"

    assert calls, "the projection has to run"
    assert set(calls) == {("cpu", "cpu")}, "the host Engram must not hand the projection an accelerator tensor"


def test_the_tiled_cpu_path_is_unchanged_by_the_offload(monkeypatch) -> None:
    """`forward` and the tiled span agree at several tile widths, on the host, bit for bit."""
    engram = _engram()
    x, ids, mask = _tokens(engram, tokens=10)
    mask[:, 3:7] = False
    monkeypatch.setattr(modules_module, "HC_TOKEN_TILE", 0)
    whole = engram(x, ids, mask)
    for tile in (3, 4, 11):
        monkeypatch.setattr(modules_module, "HC_TOKEN_TILE", tile)
        assert torch.equal(engram(x, ids, mask), whole), f"tile {tile}"


# ------------------------------------------------------------------------------------------------
# The boundary itself, without an accelerator
# ------------------------------------------------------------------------------------------------


class _RemoteTensor:
    """A tensor-shaped stand-in carrying a device object torch cannot allocate on.

    `Engram.forward`'s offload boundary does three things to its inputs -- read `.device`, `.to(host)`
    and, for the result, `.to(the caller's device)` -- and that is all this needs to model. The
    arithmetic is replaced by `_forward_span` in the tests below, which is the point: this case is
    about which device each operand is on when the arithmetic starts, not about the arithmetic.
    """

    def __init__(self, data: torch.Tensor, device) -> None:
        self.data = data
        self.device = device

    @property
    def shape(self) -> torch.Size:
        return self.data.shape

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    def to(self, where):
        if isinstance(where, torch.device):
            return _RemoteTensor(self.data.to(where), where)
        # A stand-in target: the boundary is expected to hand the result back to whatever device the
        # caller's activation was on, and the test asserts that below by identity.
        return _RemoteTensor(self.data, where)


def test_the_offload_boundary_moves_the_span_to_the_host_and_the_result_back(monkeypatch) -> None:
    """Every operand reaches the arithmetic on the host; the result returns to the caller's device."""
    engram = _engram()
    assert engram.wkv.weight.device.type == "cpu"
    calls: dict[str, object] = {}

    def span(x, hash_ids, token_mask):
        calls["x"], calls["ids"], calls["mask"] = x, hash_ids, token_mask
        # Host result of the unchanged host arithmetic. It is wrapped so that the second half of the
        # boundary -- carrying it back to a device torch cannot allocate on -- is observable here.
        calls["result"] = torch.zeros(x.shape, dtype=x.dtype)
        return _RemoteTensor(calls["result"], torch.device("cpu"))

    monkeypatch.setattr(engram, "_forward_span", span)
    x = _RemoteTensor(torch.zeros(1, 4, HC_MULT, DIM, dtype=torch.bfloat16), _FAKE)
    ids = _RemoteTensor(torch.zeros(1, 4, _hash_columns(), dtype=torch.int64), _FAKE)
    mask = _RemoteTensor(torch.ones(1, 4, dtype=torch.bool), _FAKE)

    result = engram(x, ids, mask)

    for name in ("x", "ids", "mask"):
        assert calls[name].device.type == "cpu", f"{name} must be carried to the host"
    assert torch.equal(calls["x"].data, x.data), "the span crosses unchanged"
    assert calls["ids"].data.dtype == torch.int64 and calls["mask"].data.dtype == torch.bool
    # The result is carried back to the device the caller handed in -- not to a hard-coded card.
    assert result.device is _FAKE
    assert result.data is calls["result"]


def test_a_none_mask_stays_none_across_the_boundary(monkeypatch) -> None:
    """The mask is optional, and a text-only decode must not materialize one on the way across."""
    engram = _engram()
    calls: dict[str, object] = {}

    def span(x, hash_ids, token_mask):
        calls["mask"] = token_mask
        return _RemoteTensor(torch.zeros(x.shape, dtype=x.dtype), torch.device("cpu"))

    monkeypatch.setattr(engram, "_forward_span", span)
    x = _RemoteTensor(torch.zeros(1, 2, HC_MULT, DIM, dtype=torch.bfloat16), _FAKE)
    ids = _RemoteTensor(torch.zeros(1, 2, _hash_columns(), dtype=torch.int64), _FAKE)
    engram(x, ids)
    assert calls["mask"] is None


def test_a_cuda_block_with_a_host_engram_is_not_silently_bridged(monkeypatch) -> None:
    """The bridge is for Ascend; any other accelerator keeps the old construction mismatch.

    A `cuda` device object constructs without a driver, so this asserts the *decision*: the pass is
    handed the caller's own operands rather than a host copy, and the mismatch stays visible.
    """
    engram = _engram()
    cuda = torch.device("cuda", 0)
    reached: dict[str, object] = {}

    def span(x, hash_ids, token_mask):
        reached["x"] = x
        return torch.zeros(x.shape, dtype=x.dtype)

    monkeypatch.setattr(engram, "_forward_span", span)
    x = _RemoteTensor(torch.zeros(1, 1, HC_MULT, DIM, dtype=torch.bfloat16), cuda)
    ids = _RemoteTensor(torch.zeros(1, 1, _hash_columns(), dtype=torch.int64), cuda)
    engram(x, ids)
    assert reached["x"] is x, "a CUDA stream must take the unchanged direct path"


# ------------------------------------------------------------------------------------------------
# End to end, on a leased card
# ------------------------------------------------------------------------------------------------


def test_a_real_npu_engram_matches_its_cpu_oracle(monkeypatch) -> None:
    """Opt-in. Requires a leased Ascend device; the lead runs this, the suite skips it.

    Asserts the four things the requirement is about: every parameter is on the host, the table
    gather and the projection's activation are on the host, the returned tensor is BF16 on the card,
    and its bits are the CPU run's.
    """
    device_name = os.environ.get("RELIC_ASCEND_ENGRAM_DEVICE")
    if not device_name:
        pytest.skip("set RELIC_ASCEND_ENGRAM_DEVICE=npu:0 to run the device case")
    torch_npu = pytest.importorskip("torch_npu")  # noqa: F401  (registers the device)
    device = torch.device(device_name)

    engram = _engram()
    x, ids, mask = _tokens(engram, tokens=3)
    expected = engram(x, ids, mask)
    assert expected.device.type == "cpu" and expected.dtype == torch.bfloat16

    observed: dict[str, torch.device] = {}
    table_lookup = engram.embed.lookup

    def spy_lookup(indices, device_arg=None):
        observed["ids"] = indices.device
        observed["lookup_device_arg"] = device_arg
        return table_lookup(indices, device_arg)

    engram.embed.lookup = spy_lookup
    forward_pass = engram._forward_pass

    def spy_pass(host_x, host_ids, host_mask):
        observed["pass_x"] = host_x.device
        observed["pass_ids"] = host_ids.device
        if host_mask is not None:
            observed["pass_mask"] = host_mask.device
        return forward_pass(host_x, host_ids, host_mask)

    monkeypatch.setattr(engram, "_forward_pass", spy_pass)

    result = engram(x.to(device), ids.to(device), mask.to(device))
    torch.npu.synchronize(device)

    assert result.device.type == "npu" and result.device.index == device.index
    assert result.dtype == torch.bfloat16
    assert all(parameter.device.type == "cpu" for parameter in engram.parameters())
    assert observed["ids"].type == "cpu", "the table gather must run where the table is"
    assert observed["lookup_device_arg"].type == "cpu"
    assert observed["pass_x"].type == "cpu", "the projection and gate run on the host"
    assert observed["pass_ids"].type == "cpu"
    assert observed["pass_mask"].type == "cpu"
    assert torch.equal(result.cpu(), expected), "the host arithmetic is the CPU run's arithmetic"


@pytest.mark.parametrize("device", ["npu", "npu:0"])
def test_npu_resident_engram_is_refused_before_any_bank_or_table_allocation(device, monkeypatch):
    from relicllm.models.deepseek_v4_1 import loader

    def unexpected_bank_access():
        pytest.fail("the resident-Engram refusal must happen before reading or allocating a bank")

    monkeypatch.setattr(loader.resident_bank, "enabled", unexpected_bank_access)
    # Invalid sentinels deliberately prove that no checkpoint/config is accessed
    # and no NPU is initialized before this unsupported placement is refused.
    with pytest.raises(ValueError, match="--resident-engram.*NPU backbones"):
        loader.load_backbone(None, None, device=device, resident_engram=True)
