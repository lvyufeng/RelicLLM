"""Rank-zero execution of the host Engram, with the result broadcast as exact bytes.

On Ascend the whole Engram runs on the host -- table, hash lookup, ``wkv`` projection and gate
(``modules._engram_device``) -- and a tensor-parallel run repeats that identical host work on every
rank: the measured decode cost is about **1.803 s/token across eight ranks**, seven eighths of it
redundant, because the Engram is replicated and its inputs are identical under TP. This module lets
one rank do the host computation and hands the *result bytes* to the others.

The correctness contract is deliberately narrow:

* **Nothing about the computation moves.** The caller passes its existing host body
  (``Engram._forward_span``); it runs on the host with the original weights, the original source-code
  metadata and the original reduction. No dense approximation, no quantization change.
* **The first call per Engram module is proved, not assumed.** Every rank computes the host body
  once, and the raw input, hash ids, mask and result bytes are compared across all ranks. After that
  proof only rank 0 computes; the others receive.
* **The wire carries bytes, never arithmetic.** The preferred wire is an ``int8`` byte alias of the
  BF16 result, so no FP32 node can flush a subnormal and no BF16 arithmetic can re-round a value.
  ``wire="bf16"`` is an exact alternative when a backend cannot reinterpret dtypes: the collective
  still only moves the elements' own bytes.
* **Nothing here is bound until the root asks.** ``enabled_from_environment`` reads
  ``DEEPSEEK_V41_ENGRAM_RANK0`` (default off). The root attaches one :class:`HostEngramState` per
  Engram, so two models in one process never share a proof or a buffer.

This file makes no NPU-graph claim, does not touch the checkpoint, does not move the table, and does
not change any quantization or reduction. A real multi-rank run is the root's to execute: this module
ships a standalone probe (:func:`main`) that exercises the collectives on eight ranks without a
checkpoint.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os

import torch

__all__ = [
    "ENGRAM_RANK0_ENV",
    "ENGRAM_VERIFY_ALWAYS_ENV",
    "HostEngramState",
    "enabled_from_environment",
    "rank0_host_engram",
    "verify_always_from_environment",
]

#: Set to a true spelling to bind the rank-zero host Engram (default off).
ENGRAM_RANK0_ENV = "DEEPSEEK_V41_ENGRAM_RANK0"
#: Set to a true spelling to re-run the cross-rank proof on *every* call (debug only).
ENGRAM_VERIFY_ALWAYS_ENV = "DEEPSEEK_V41_ENGRAM_RANK0_VERIFY_ALWAYS"

_TRUE = frozenset({"1", "true", "yes", "on"})
_DIGEST_SIZE = 32
#: One token's BF16 Engram result at the released geometry: hc_mult * dim * 2 bytes ~= 40 KiB. A
#: payload at or under this is compared as raw bytes; a larger one is compared by SHA-256 digest, which
#: still covers every byte. Either way one collective carries at most ``world * 40 KiB``.
_RAW_VERIFY_BYTES = 40 * 1024


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE


def enabled_from_environment() -> bool:
    """Whether ``DEEPSEEK_V41_ENGRAM_RANK0`` asks for the rank-zero path (default ``False``)."""
    return _flag(ENGRAM_RANK0_ENV)


def verify_always_from_environment() -> bool:
    """Whether every call re-runs the proof (debug only; default ``False``)."""
    return _flag(ENGRAM_VERIFY_ALWAYS_ENV)


def _collective_device() -> torch.device:
    """The device a collective must carry its tensors on: the card this process drives, else the host.

    Reading the current device is not setting one; this module never calls ``set_device``. The
    payload of a proof collective is tiny (a 128-byte digest, or one result span of at most 40 KiB),
    so moving it here and back for the byte comparison is not a cost worth avoiding.
    """
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu", torch.npu.current_device())
    return torch.device("cpu")


def _device_type(tensor: torch.Tensor) -> str:
    """The device type the dispatch decides on.

    A seam rather than a placement: portable tests patch this to exercise the distributed branch with
    host tensors, and every operation below still uses the tensor's real device, so the code under
    test is the code that runs.
    """
    return tensor.device.type


def _host_bytes(tensor: torch.Tensor) -> bytes:
    """``tensor``'s exact storage bytes on the host, for comparison or for the wire."""
    return tensor.detach().contiguous().view(torch.int8).cpu().numpy().tobytes()


def _digest(name: str, tensor: torch.Tensor | None, *, error: bool = False,
            data: bytes | None = None) -> bytes:
    """A digest of one payload's *bytes*, shape, dtype and presence, so absence is distinguishable.

    ``data`` lets the caller hand in bytes it has already materialized, because the one-time proof
    reads each payload twice (digest and raw comparison) and a large span should be copied once.
    """
    digest = hashlib.sha256()
    digest.update(name.encode("ascii"))
    if error:
        digest.update(b"\xfferror")
    elif tensor is None:
        digest.update(b"\xffnone")
    else:
        digest.update(b"\xffpresent")
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(_host_bytes(tensor) if data is None else data)
    return digest.digest()


def _gather_rows(state: HostEngramState, local: torch.Tensor) -> torch.Tensor:
    """All-gather one equal-size ``int8`` vector from every rank; returns ``[world, numel]``."""
    state.gathers += 1
    parts = state.gather(local.contiguous())
    rows = [part.detach().contiguous().view(torch.int8).cpu() for part in parts]
    if len(rows) != state.world:
        raise RuntimeError(
            f"the host Engram gathered {len(rows)} rows from a world of {state.world}"
        )
    width = int(local.numel())
    if any(int(row.numel()) != width for row in rows):
        raise RuntimeError(
            "the host Engram collective payloads have different sizes across ranks "
            f"({[int(row.numel()) for row in rows]} vs {width}); the ranks do not agree on the span"
        )
    state.bytes_gathered += width * state.world
    return torch.stack(rows)


def _expect_same_bytes(state: HostEngramState, name: str, data: bytes) -> None:
    """Require this rank's raw payload bytes to equal every other rank's."""
    local = torch.frombuffer(bytearray(data), dtype=torch.int8)
    rows = _gather_rows(state, local)
    first = rows[0]
    differing = [rank for rank in range(rows.size(0)) if not bool(torch.equal(rows[rank], first))]
    if differing:
        raise RuntimeError(
            f"the host Engram payload {name!r} differs on ranks {differing}: every rank must see the "
            "same activation, hash ids and mask under tensor parallelism; refusing to serve results "
            "computed from one rank's inputs"
        )


def _checked_result(result, reference: torch.Tensor) -> torch.Tensor:
    """The host body's result, required to be a contiguous host tensor shaped like its input."""
    if not isinstance(result, torch.Tensor):
        raise TypeError(f"the host Engram body returned {type(result).__name__}, not a tensor")
    if result.device.type != "cpu":
        raise ValueError(
            f"the host Engram body must return a host tensor, got {result.device}; the whole point "
            "of this path is that the projection and gate stay on the CPU"
        )
    if result.dtype != reference.dtype:
        raise ValueError(
            f"the host Engram body returned {result.dtype} for a {reference.dtype} activation; the "
            "result is carried as exact bytes, so its width has to match"
        )
    if tuple(result.shape) != tuple(reference.shape):
        raise ValueError(
            f"the host Engram body returned shape {tuple(result.shape)} for an activation of shape "
            f"{tuple(reference.shape)}; the Engram writes the residual back in the input's shape"
        )
    return result.detach().contiguous()


class HostEngramState:
    """One Engram module's rank-zero state: the proof, the wire and the counters.

    The root builds one of these per Engram (never a module-level cache, so two models in one process
    cannot share a proof) and attaches it to that module. :meth:`call` is the whole per-call contract;
    :func:`rank0_host_engram` is the module-level spelling of the same thing.

    ``world == 1`` or a non-NPU activation means the helper is a pass-through to the caller's host
    body, so CPU, CUDA and single-rank runs keep the arithmetic they had. Distributed use requires an
    initialized process group (:meth:`from_distributed`) and injected collectives in tests.
    """

    __slots__ = (
        "backend",
        "broadcast",
        "broadcasts",
        "bytes_broadcast",
        "bytes_gathered",
        "calls",
        "dtype",
        "gather",
        "gathers",
        "host_computes",
        "rank",
        "raw_verify_bytes",
        "verification",
        "verified",
        "verified_calls",
        "verify_always",
        "wire",
        "world",
    )

    def __init__(self, rank: int, world: int, *, gather=None, broadcast=None,
                 wire: str = "int8", backend: str | None = None, verify_always: bool = False,
                 raw_verify_bytes: int = _RAW_VERIFY_BYTES):
        if isinstance(rank, bool) or isinstance(world, bool):
            raise TypeError("rank and world must be integers")
        if not isinstance(rank, int) or not isinstance(world, int):
            raise TypeError("rank and world must be integers")
        if world < 1:
            raise ValueError(f"world must be >= 1, got {world}")
        if not 0 <= rank < world:
            raise ValueError(f"rank {rank} is outside a world of {world}")
        if wire not in {"int8", "bf16"}:
            raise ValueError(f"wire must be 'int8' or 'bf16', got {wire!r}")
        if isinstance(raw_verify_bytes, bool) or not isinstance(raw_verify_bytes, int):
            raise TypeError("raw_verify_bytes must be an int")
        if raw_verify_bytes < 0:
            raise ValueError("raw_verify_bytes must be >= 0")
        if world > 1 and (gather is None or broadcast is None):
            raise ValueError(
                "a world > 1 host Engram state needs injected `gather` and `broadcast` callables; "
                "construct it with HostEngramState.from_distributed() instead"
            )
        self.rank = rank
        self.world = world
        self.gather = gather
        self.broadcast = broadcast
        self.wire = wire
        self.backend = backend
        self.verify_always = bool(verify_always)
        self.raw_verify_bytes = raw_verify_bytes
        self.verified = False
        self.dtype: torch.dtype | None = None
        self.calls = 0
        self.host_computes = 0
        self.gathers = 0
        self.broadcasts = 0
        self.bytes_gathered = 0
        self.bytes_broadcast = 0
        self.verified_calls = 0
        self.verification: dict[str, object] = {}

    @classmethod
    def from_distributed(cls, *, wire: str = "int8", verify_always: bool | None = None,
                         device: torch.device | str | None = None) -> HostEngramState:
        """The state for the process group this rank is already in.

        Refuses an uninitialized group by name: a silently local fallback would look like the
        deduplication working while every rank still ran the whole host Engram.

        ``device`` is where the *collectives* carry their tensors -- the caller's card. The proof
        payload is built on the host (it is a digest, or a span the host body just produced), moved
        there for the collective and brought back for the byte comparison. ``None`` resolves to this
        process's current NPU when there is one, and to the host otherwise.
        """
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "the rank-zero host Engram needs an initialized torch.distributed process group; "
                "none is initialized in this process"
            )
        world, rank, backend = dist.get_world_size(), dist.get_rank(), dist.get_backend()
        target = _collective_device() if device is None else torch.device(device)

        def gather(local: torch.Tensor):
            payload = local.contiguous()
            if payload.device != target:
                payload = payload.to(target)
            parts = [torch.empty_like(payload) for _ in range(world)]
            dist.all_gather(parts, payload)
            return parts

        def broadcast(tensor: torch.Tensor, src: int = 0):
            dist.broadcast(tensor, src=src)
            return tensor

        return cls(rank, world, gather=gather, broadcast=broadcast, wire=wire, backend=backend,
                   verify_always=verify_always_from_environment()
                   if verify_always is None else verify_always)

    # -- the per-call contract --------------------------------------------------------------------

    def call(self, host_compute, x: torch.Tensor, hash_ids: torch.Tensor,
             token_mask: torch.Tensor | None) -> torch.Tensor:
        """Run ``host_compute`` on rank 0 and return its exact result on every rank's device."""
        if not callable(host_compute):
            raise TypeError("host_compute must be the callable that runs the Engram on the host")
        on_device = _device_type(x) == "npu"
        if self.world < 2:
            if not on_device:
                return host_compute(x, hash_ids, token_mask)
            # Bound with one rank: keep the old host boundary rather than handing a host body an
            # accelerator tensor. Nothing is deduplicated and no proof is claimed.
            host_x = x.detach().contiguous().to("cpu")
            host_ids = hash_ids.detach().contiguous().to("cpu")
            host_mask = None if token_mask is None else token_mask.detach().contiguous().to("cpu")
            moved = _checked_result(host_compute(host_x, host_ids, host_mask), host_x)
            return moved.to(x.device)
        if not on_device:
            return host_compute(x, hash_ids, token_mask)
        if x.dtype != torch.bfloat16:
            raise ValueError(
                f"the rank-zero host Engram carries a bfloat16 result as exact bytes, and this "
                f"activation is {x.dtype}"
            )
        if self.dtype is not None and x.dtype != self.dtype:
            raise ValueError(
                f"the host Engram was proved on {self.dtype} and this call is {x.dtype}; the wire "
                "carries the result's bytes and has to know the element width"
            )
        self.calls += 1
        host_x = x.detach().contiguous().to("cpu")
        host_ids = hash_ids.detach().contiguous().to("cpu")
        host_mask = None if token_mask is None else token_mask.detach().contiguous().to("cpu")

        computed = None
        if self.verify_always or not self.verified:
            computed = self._verify(host_compute, host_x, host_ids, host_mask)
            self.verified = True
            self.verified_calls += 1
        if self.rank == 0:
            result = computed if computed is not None else host_compute(host_x, host_ids, host_mask)
            result = _checked_result(result, host_x)
            self.dtype = result.dtype
            self.host_computes += 1
        else:
            result = None
        return self._broadcast(x, result)

    def _verify(self, host_compute, host_x, host_ids, host_mask) -> torch.Tensor | None:
        """Every rank computes once and the four payloads are compared across the world."""
        computed, failure = None, None
        try:
            # Validation is inside the try on purpose: a rank whose host body returns the wrong
            # shape or dtype must still reach the collective, or the other ranks would wait for a
            # payload that never comes. Folding it in reports such a rank as a payload mismatch.
            computed = _checked_result(host_compute(host_x, host_ids, host_mask), host_x)
        except BaseException as error:  # noqa: BLE001 - a failing rank still joins the collective
            failure = error
        self.dtype = host_x.dtype
        payloads = (("x", host_x), ("hash_ids", host_ids), ("mask", host_mask), ("result", computed))
        blobs: dict[str, bytes] = {}
        digests = []
        for name, tensor in payloads:
            if tensor is not None:
                blobs[name] = _host_bytes(tensor)
            digests.append(
                _digest(name, tensor, error=(name == "result" and failure is not None),
                        data=blobs.get(name))
            )
        rows = _gather_rows(self, torch.frombuffer(bytearray(b"".join(digests)), dtype=torch.int8))
        if failure is not None:
            # The failing rank reports what actually broke; its peers see the result payload's
            # sentinel digest and report the mismatch. Both leave the proof, and neither waits.
            raise failure
        for index, (name, _) in enumerate(payloads):
            column = rows[:, index * _DIGEST_SIZE:(index + 1) * _DIGEST_SIZE]
            differing = [rank for rank in range(rows.size(0))
                         if not bool(torch.equal(column[rank], column[0]))]
            if differing:
                raise RuntimeError(
                    f"the host Engram payload {name!r} differs on ranks {differing} (by digest): "
                    "every rank must compute the Engram from the same activation, hash ids and mask; "
                    "refusing to replace seven of those computations with one rank's result"
                )
        raw, digested = [], []
        for name, tensor in payloads:
            if tensor is None:
                continue
            data = blobs[name]
            if 0 < len(data) <= self.raw_verify_bytes:
                _expect_same_bytes(self, name, data)
                raw.append(name)
            else:
                digested.append(name)
        self.verification = {
            "raw": raw,
            "digest": digested,
            "bytes_per_rank": {name: len(data) for name, data in blobs.items()},
            "shape": list(host_x.shape),
            "dtype": str(host_x.dtype),
        }
        return computed

    def _broadcast(self, x: torch.Tensor, result: torch.Tensor | None) -> torch.Tensor:
        """Publish rank 0's exact bytes to every rank, on that rank's own caller device."""
        shape = tuple(x.shape) if result is None else tuple(result.shape)
        dtype = x.dtype if result is None else result.dtype
        if self.wire == "int8":
            count = 1
            for size in shape:
                count *= int(size)
            count *= torch.empty((), dtype=dtype).element_size()
            if self.rank == 0:
                wire = torch.frombuffer(bytearray(_host_bytes(result)), dtype=torch.int8).clone()
                wire = wire.to(x.device)
            else:
                wire = torch.empty(count, dtype=torch.int8, device=x.device)
            self._publish(wire)
            try:
                return wire.view(dtype).view(shape).contiguous()
            except RuntimeError as error:
                raise RuntimeError(
                    "this backend cannot reinterpret the int8 wire as the result dtype; build the "
                    "state with wire='bf16', which still moves only the elements' own bytes"
                ) from error
        if self.rank == 0:
            published = result.to(x.device)
        else:
            published = torch.empty(shape, dtype=dtype, device=x.device)
        self._publish(published)
        return published.contiguous()

    def _publish(self, tensor: torch.Tensor) -> None:
        if tensor.numel() == 0:
            return
        self.broadcasts += 1
        self.bytes_broadcast += int(tensor.numel())
        self.broadcast(tensor, 0)

    def describe(self) -> dict[str, object]:
        """Counters for the launcher's log; no tensor, shape-independent."""
        return {
            "rank": self.rank, "world": self.world, "wire": self.wire, "backend": self.backend,
            "verified": self.verified, "verify_always": self.verify_always,
            "calls": self.calls, "verified_calls": self.verified_calls,
            "host_computes": self.host_computes, "gathers": self.gathers,
            "broadcasts": self.broadcasts, "bytes_gathered": self.bytes_gathered,
            "bytes_broadcast": self.bytes_broadcast, "verification": dict(self.verification),
        }


def rank0_host_engram(host_compute, x: torch.Tensor, hash_ids: torch.Tensor,
                      token_mask: torch.Tensor | None, state: HostEngramState | None
                      ) -> torch.Tensor:
    """Run the host Engram on rank 0 and give every rank its exact result.

    ``state`` is the module's :class:`HostEngramState`; ``None`` (or a world of one) runs
    ``host_compute(x, hash_ids, token_mask)`` unchanged, which is what the caller had before.
    """
    if state is None:
        return host_compute(x, hash_ids, token_mask)
    return state.call(host_compute, x, hash_ids, token_mask)


# ------------------------------------------------------------------------------------------------
# Standalone 8-rank probe (no checkpoint): `python -m torch.distributed.run ... -m <this module>`
# ------------------------------------------------------------------------------------------------

def _probe_host_compute(seed: int, dim: int, hc_mult: int, hash_cols: int):
    """A deterministic host body with the Engram's shape: lookup, projection, gate, BF16 narrowing."""
    generator = torch.Generator().manual_seed(seed)
    rows, head_dim = 4096, 16
    table = torch.randn(rows, head_dim, generator=generator).to(torch.float32)
    wkv = torch.randn(hash_cols * head_dim, dim * (hc_mult + 1), generator=generator) * 0.02
    eps, clamp_value = 1e-6, 1e-6

    def body(x: torch.Tensor, hash_ids: torch.Tensor, token_mask: torch.Tensor | None):
        del token_mask  # the probe's mask only exercises the wire, not the gate
        gathered = table[hash_ids.reshape(-1).long()].reshape(*hash_ids.shape[:-1], -1)
        kv = gathered.float() @ wkv
        key, value = kv.split([hc_mult * dim, dim], dim=-1)
        key = key.unflatten(-1, (hc_mult, dim))
        h = x.float()
        rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
        dot = (h * key).sum(-1) * rstd * dim**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(clamp_value).sqrt(), dot))
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)

    return body


def main(argv: list[str] | None = None) -> int:
    """Run the probe on every rank of an existing (or freshly started) process group."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--hc-mult", type=int, default=2)
    parser.add_argument("--hash-cols", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--backend", default=None, help="process-group backend when starting one")
    parser.add_argument("--device", default=None, help="override the per-rank device")
    args = parser.parse_args(argv)

    started = dt.datetime.now(dt.timezone.utc)
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        if args.backend is None:
            parser.error("no initialized process group: run under torch.distributed.run or pass "
                         "--backend")
        dist.init_process_group(backend=args.backend, timeout=dt.timedelta(minutes=5))
        owns_group = True
    else:
        owns_group = False
    world, rank = dist.get_world_size(), dist.get_rank()

    if args.device:
        device = torch.device(args.device)
    elif hasattr(torch, "npu") and torch.npu.is_available():
        device = torch.device("npu", rank % torch.npu.device_count())
    else:
        device = torch.device("cpu")

    body = _probe_host_compute(seed=17, dim=args.dim, hc_mult=args.hc_mult,
                               hash_cols=args.hash_cols)
    generator = torch.Generator().manual_seed(41)
    host_x = torch.randn(1, args.tokens, args.hc_mult, args.dim,
                         generator=generator).to(torch.bfloat16)
    host_ids = torch.randint(0, 4096, (1, args.tokens, args.hash_cols), generator=generator)
    host_mask = torch.ones(1, args.tokens, dtype=torch.bool)

    state = HostEngramState.from_distributed(device=device)
    x = host_x.to(device)
    ids = host_ids.to(device) if device.type != "cpu" else host_ids
    mask = host_mask.to(device) if device.type != "cpu" else host_mask

    observed = []
    for iteration in range(args.iterations):
        if device.type == "npu":
            torch.npu.synchronize(device)
        began = dt.datetime.now(dt.timezone.utc)
        result = rank0_host_engram(body, x, ids, mask, state)
        if device.type == "npu":
            torch.npu.synchronize(device)
        elapsed = (dt.datetime.now(dt.timezone.utc) - began).total_seconds()
        reference = body(host_x, host_ids, host_mask)
        exact = bool(torch.equal(result.cpu().view(torch.int8),
                                 reference.view(torch.int8)))
        if not exact:
            raise SystemExit(
                f"rank {rank} iteration {iteration}: the broadcast result is not bit-identical to "
                "the host computation it replaces"
            )
        observed.append({"iteration": iteration, "seconds": elapsed})

    report = {
        "started": started.isoformat(), "world": world, "rank": rank, "device": str(device),
        "wire": state.wire, "state": state.describe(), "iterations": observed,
        "first_call_seconds": observed[0]["seconds"],
        "later_call_seconds": observed[-1]["seconds"] if len(observed) > 1 else None,
        "result_bits_identical_to_local_host_compute": True,
        "own_process_group": owns_group,
    }
    if rank == 0:
        print(dt.datetime.now(dt.timezone.utc).isoformat(), "PROBE", report, flush=True)
    else:
        print(f"[rank {rank}] probe ok: {observed[-1]['seconds']:.6f} s/call", flush=True)
    if owns_group:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())