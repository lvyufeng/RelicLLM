"""The rank-zero host Engram: one host computation, exact-byte result broadcast.

On Ascend every rank runs the whole host Engram, which the profile measured at **1.803 s/token
across eight ranks** -- seven eighths of it redundant, because the Engram is replicated and its
inputs are identical under TP. `ascend_host_engram` runs the caller's host body on rank 0 and
broadcasts the result's own bytes.

What these tests pin:

* the **first call per module is proved**: every rank computes, and the raw activation, hash ids,
  mask and result bytes are compared across all ranks (raw bytes for payloads at or under 40 KiB,
  a SHA-256 digest of the whole payload above that), with a mismatch refused by name;
* after the proof **only rank 0 computes** -- asserted through a per-rank call counter, not inferred;
* the wire is **bytes, not arithmetic**: `int8` alias by default, `bf16` as an exact alternative, and
  signed zeros and BF16 subnormals survive bit for bit;
* CPU/single-rank callers keep the body they had, and a world > 1 state without collectives, an
  uninitialized process group, or a malformed rank/world is refused rather than degraded.

The multi-rank cases run against a threaded stand-in for a process group: real barrier semantics,
real payload bytes, no accelerator. `ascend_host_engram.main` is the probe the lead runs on eight
real ranks; it is exercised here end to end on a one-rank `gloo` group, which is the same code path
minus the broadcast fan-out.

The one thing these tests cannot show is the real HCCL collective moving int8 on an Ascend card, and
they do not claim it: the gated case only proves the byte-alias primitive on the device the lead
leases, and the probe is the multi-rank check.
"""

from __future__ import annotations

import os
import threading
from collections import Counter

import pytest
import torch

from relicllm.models.deepseek_v4_1 import ascend_host_engram as host_engram
from relicllm.models.deepseek_v4_1.ascend_host_engram import (
    ENGRAM_RANK0_ENV,
    ENGRAM_VERIFY_ALWAYS_ENV,
    HostEngramState,
    enabled_from_environment,
    rank0_host_engram,
    verify_always_from_environment,
)

DIM = 8
HC_MULT = 2
COLS = 4
ROWS = 32
DEVICE_ENV = "RELIC_ASCEND_HOST_ENGRAM_DEVICE"

# ------------------------------------------------------------------------------------------------
# A threaded stand-in for a process group
# ------------------------------------------------------------------------------------------------


class FakeWorld:
    """One process group in one process: each rank runs in its own thread.

    `gather` and `broadcast` follow the collective contract the helper relies on -- every rank calls
    each one, equal payload sizes, source rank 0 for the broadcast -- and synchronize with real
    barriers, so a rank that skips or reorders a collective fails here as a hang rather than passing
    silently. Payloads are the tensors' own bytes.
    """

    def __init__(self, world: int, timeout: float = 30.0):
        self.world = world
        self._barrier = threading.Barrier(world, timeout=timeout)
        self._lock = threading.Lock()
        self._slots: dict[str, dict[int, torch.Tensor]] = {}
        self.counters: Counter = Counter()
        self.bytes: Counter = Counter()
        self.log: list[tuple[str, int, int]] = []

    def collectives(self, rank: int):
        world = self

        def gather(local: torch.Tensor):
            world._arrive("gather", rank, local)
            with world._lock:
                out = [world._slots["gather"][other].clone() for other in range(world.world)]
            world._leave()
            return out

        def broadcast(tensor: torch.Tensor, src: int = 0):
            world._arrive("broadcast", rank, tensor)
            with world._lock:
                source = world._slots["broadcast"][src]
            tensor.copy_(source)
            world._leave()
            return tensor

        return gather, broadcast

    def _arrive(self, kind: str, rank: int, tensor: torch.Tensor) -> None:
        with self._lock:
            self._slots.setdefault(kind, {})[rank] = tensor.detach().clone()
            self.counters[kind] += 1
            self.bytes[kind] += int(tensor.numel())
            self.log.append((kind, rank, int(tensor.numel())))
        self._barrier.wait()

    def _leave(self) -> None:
        self._barrier.wait()


def run_ranks(world: FakeWorld, rank_body, timeout: float = 60.0):
    """Run `rank_body(rank)` on one thread per rank and return `(results, errors)`."""
    results: dict[int, object] = {}
    errors: dict[int, BaseException] = {}

    def target(rank: int) -> None:
        try:
            results[rank] = rank_body(rank)
        except BaseException as error:  # noqa: BLE001 - reported per rank
            errors[rank] = error

    threads = [threading.Thread(target=target, args=(rank,), daemon=True)
               for rank in range(world.world)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout)
    assert not any(thread.is_alive() for thread in threads), "a rank hung inside a collective"
    return results, errors


@pytest.fixture
def npu_dispatch(monkeypatch):
    """Make host tensors take the distributed NPU branch.

    This patches only the dispatch predicate; the payload copies, the collectives, the byte alias and
    the result construction are all the production ones, running on real (host) tensors.
    """
    monkeypatch.setattr(host_engram, "_device_type", lambda tensor: "npu")


# ------------------------------------------------------------------------------------------------
# The body under test
# ------------------------------------------------------------------------------------------------


def make_body(seed: int = 0):
    """A deterministic host body: table lookup, FP32 gate over a projection, one BF16 narrowing."""
    generator = torch.Generator().manual_seed(seed)
    table = torch.randn(ROWS, 4, generator=generator)
    width = COLS * 4
    weight = torch.randn(width, DIM, generator=generator) * 0.05

    def body(x: torch.Tensor, hash_ids: torch.Tensor, token_mask: torch.Tensor | None):
        assert x.device.type == "cpu", "the host body must run on the host"
        picked = table[hash_ids.long()].flatten(-2).float()
        key = picked @ weight
        h = x.float()
        rstd = torch.rsqrt(h.square().mean(-1, keepdim=True) + 1e-6)
        dot = (h * key.unsqueeze(-2)).sum(-1, keepdim=True) * rstd * DIM**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        out = h + gate * torch.sin(h)
        if token_mask is not None:
            out = torch.where(token_mask[:, :, None, None], out, h)
        return out.to(x.dtype)

    return body


def inputs(batch: int = 1, tokens: int = 1, seed: int = 5):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, tokens, HC_MULT, DIM, generator=generator).to(torch.bfloat16)
    ids = torch.randint(0, ROWS, (batch, tokens, COLS), generator=generator)
    mask = torch.ones(batch, tokens, dtype=torch.bool)
    return x, ids, mask


def bits(tensor: torch.Tensor) -> torch.Tensor:
    """The tensor's exact bytes as an int8 tensor, for bit comparisons."""
    return tensor.detach().contiguous().view(torch.int8)


def counting(body, per_rank_computes: Counter, rank: int, *, perturb=None):
    """Wrap `body`, counting its calls per rank and optionally perturbing one rank's result."""

    def wrapped(x, hash_ids, token_mask):
        per_rank_computes[rank] += 1
        result = body(x, hash_ids, token_mask)
        if perturb is not None:
            result = perturb(result)
        return result

    return wrapped


def one_call_per_rank(world: FakeWorld, *, body_factory=make_body, state_kwargs=None,
                      calls: int = 1, x_override=None, perturb_result=None):
    """Build one state per rank, run `calls` calls on every rank, return `(results, errors, states, computes)`."""
    computes: Counter = Counter()
    states: dict[int, HostEngramState] = {}
    kwargs = dict(state_kwargs or {})

    def rank_body(rank: int):
        gather, broadcast = world.collectives(rank)
        state = HostEngramState(rank, world.world, gather=gather, broadcast=broadcast, **kwargs)
        states[rank] = state
        base = body_factory(rank) if body_factory is not make_body else make_body()
        x, ids, mask = inputs()
        if x_override is not None and rank in x_override:
            x = x_override[rank](x)
        wrapped = counting(base, computes, rank,
                           perturb=(None if perturb_result is None else perturb_result.get(rank)))
        produced = [rank0_host_engram(wrapped, x, ids, mask, state) for _ in range(calls)]
        return produced

    results, errors = run_ranks(world, rank_body)
    return results, errors, states, computes


# ------------------------------------------------------------------------------------------------
# The fallbacks: CPU, CUDA and a single rank
# ------------------------------------------------------------------------------------------------


def test_a_host_activation_or_a_single_rank_world_runs_the_caller_body_directly():
    """No collective, no copy, no state: exactly the call the caller made before this module."""
    seen: list[tuple[str, tuple[int, ...]]] = []

    def body(x, hash_ids, token_mask):
        seen.append((x.device.type, tuple(x.shape)))
        return x.clone()

    x, ids, mask = inputs()
    direct = rank0_host_engram(body, x, ids, mask, None)
    assert torch.equal(bits(direct), bits(x))
    single = HostEngramState(0, 1)
    fallback = rank0_host_engram(body, x, ids, mask, single)
    assert torch.equal(bits(fallback), bits(x))
    assert seen == [("cpu", tuple(x.shape)), ("cpu", tuple(x.shape))]
    # The single-rank state never verified and never reports host work it did not deduplicate.
    assert single.verified is False
    assert single.calls == 0 and single.gathers == 0 and single.broadcasts == 0


def test_a_single_rank_bound_on_a_card_still_keeps_the_host_boundary(npu_dispatch):
    """world == 1 with the flag on: the body still runs on the host, and no proof is claimed."""
    seen: list[str] = []

    def body(x, hash_ids, token_mask):
        seen.append(x.device.type)
        return x.clone()

    x, ids, mask = inputs()
    state = HostEngramState(0, 1)
    result = rank0_host_engram(body, x, ids, mask, state)
    assert seen == ["cpu"], "a host body must never be handed the card's tensor"
    assert torch.equal(bits(result), bits(x))
    assert state.verified is False and state.gathers == 0


def test_a_missing_callable_is_refused_by_name():
    state = HostEngramState(0, 1)
    with pytest.raises(TypeError, match="host_compute must be the callable"):
        rank0_host_engram(None, torch.zeros(1), torch.zeros(1), None, state)


# ------------------------------------------------------------------------------------------------
# The distributed contract
# ------------------------------------------------------------------------------------------------


def test_the_first_call_computes_on_every_rank_and_the_next_only_on_rank_zero(npu_dispatch):
    world = FakeWorld(8)
    results, errors, states, computes = one_call_per_rank(world, calls=2)
    assert not errors, errors

    # Every rank produced the same exact bytes, twice.
    for rank in range(world.world):
        first, second = results[rank]
        assert torch.equal(bits(first), bits(second))
        assert torch.equal(bits(first), bits(results[0][0]))
        assert first.dtype == torch.bfloat16 and first.shape == (1, 1, HC_MULT, DIM)

    # The proof runs once: 8 computations on the first call, rank 0 only on the second.
    assert computes == Counter({0: 2, **{rank: 1 for rank in range(1, 8)}})
    for rank in range(world.world):
        state = states[rank]
        assert state.verified is True
        assert state.verified_calls == 1
        assert state.calls == 2
        assert state.verification["shape"] == [1, 1, HC_MULT, DIM]
        assert state.verification["dtype"] == "torch.bfloat16"

    # One digest collective plus one raw collective per payload at or under the raw bound, once; then
    # one broadcast of the result per call per rank.
    assert world.counters["gather"] == world.world * 5
    assert world.counters["broadcast"] == world.world * 2
    assert states[0].verification["raw"] == ["x", "hash_ids", "mask", "result"]
    assert states[0].verification["digest"] == []
    # int8 wire: two bytes per BF16 element.
    assert world.bytes["broadcast"] == world.world * 2 * 2 * (1 * 1 * HC_MULT * DIM)


def test_a_large_payload_is_proved_by_digest_and_a_small_one_by_raw_bytes(npu_dispatch):
    """Above the 40 KiB raw bound the whole payload is still covered, by its SHA-256 digest."""
    world = FakeWorld(2)
    _, errors, states, _ = one_call_per_rank(world, state_kwargs={"raw_verify_bytes": 8})
    assert not errors, errors
    raw = states[0].verification["raw"]
    digest = states[0].verification["digest"]
    assert raw == ["mask"], "only the one-byte mask fits an 8-byte raw bound"
    assert digest == ["x", "hash_ids", "result"]
    assert states[0].verification["bytes_per_rank"]["x"] == 2 * 1 * 1 * HC_MULT * DIM


def test_a_result_shape_or_dtype_that_does_not_match_the_activation_is_refused(npu_dispatch):
    world = FakeWorld(2)

    def wrong_shape(x, hash_ids, token_mask):
        return torch.zeros(x.shape[:-1] + (DIM + 2,), dtype=x.dtype)

    _, errors, _, _ = one_call_per_rank(world, body_factory=lambda rank: wrong_shape)
    assert set(errors) == {0, 1}, "every rank must report the contract violation"
    assert all(isinstance(error, ValueError) for error in errors.values())
    assert "shape" in str(errors[0])

    def wrong_dtype(x, hash_ids, token_mask):
        return x.float()

    world = FakeWorld(2)
    _, errors, _, _ = one_call_per_rank(world, body_factory=lambda rank: wrong_dtype)
    assert set(errors) == {0, 1}
    assert "dtype" in str(errors[0]) or "torch.float32" in str(errors[0])


def test_a_divergent_activation_is_refused_and_names_the_payload_and_ranks(npu_dispatch):
    world = FakeWorld(8)

    def one_bit(x: torch.Tensor) -> torch.Tensor:
        broken = x.clone()
        broken.view(torch.int16)[0, 0, 0, 0] ^= 1
        return broken

    _, errors, _, computes = one_call_per_rank(world, x_override={3: one_bit})
    assert set(errors) == set(range(8)), "every rank must fail the same proof"
    message = str(errors[0])
    assert "'x'" in message and "[3]" in message
    assert "refusing" in message
    # The divergence is caught by the digest collective, before any rank runs a raw comparison.
    assert all(isinstance(error, RuntimeError) for error in errors.values())
    assert max(computes.values()) == 1


def test_a_divergent_hash_ids_or_mask_is_refused(npu_dispatch):
    def change_ids(ids: torch.Tensor) -> torch.Tensor:
        broken = ids.clone()
        broken[0, 0, 0] = (broken[0, 0, 0] + 1) % ROWS
        return broken

    computes: Counter = Counter()
    world = FakeWorld(4)

    def rank_body(rank: int):
        gather, broadcast = world.collectives(rank)
        state = HostEngramState(rank, world.world, gather=gather, broadcast=broadcast)
        body = counting(make_body(), computes, rank)
        x, ids, mask = inputs()
        if rank == 1:
            ids = change_ids(ids)
        return rank0_host_engram(body, x, ids, mask, state)

    _, errors = run_ranks(world, rank_body)
    assert set(errors) == set(range(4))
    assert "'hash_ids'" in str(errors[0]) and "[1]" in str(errors[0])


def test_a_none_mask_agrees_and_a_present_mask_against_none_is_refused(npu_dispatch):
    def all_none_body(rank: int):
        gather, broadcast = world.collectives(rank)
        state = HostEngramState(rank, world.world, gather=gather, broadcast=broadcast)
        x, ids, _ = inputs()
        return rank0_host_engram(counting(make_body(), Counter(), rank), x, ids, None, state)

    world = FakeWorld(4)
    _, errors = run_ranks(world, all_none_body)
    assert not errors, errors

    # One rank supplies a tensor mask, the others None: the mask payload is present on one rank and
    # absent on the rest, which the digest distinguishes.
    world = FakeWorld(4)

    def mixed_mask(rank: int):
        gather, broadcast = world.collectives(rank)
        state = HostEngramState(rank, world.world, gather=gather, broadcast=broadcast)
        x, ids, mask = inputs()
        return rank0_host_engram(counting(make_body(), Counter(), rank), x, ids,
                                 mask if rank == 2 else None, state)

    _, errors = run_ranks(world, mixed_mask)
    assert set(errors) == set(range(4))
    assert "'mask'" in str(errors[0])


def test_a_divergent_result_is_refused_by_name(npu_dispatch):
    def flip(result: torch.Tensor) -> torch.Tensor:
        broken = result.clone()
        broken.view(torch.int16)[0, 0, 0, 0] ^= 1
        return broken

    world = FakeWorld(8)
    _, errors, _, _ = one_call_per_rank(world, perturb_result={5: flip})
    assert set(errors) == set(range(8))
    assert "'result'" in str(errors[0]) and "[5]" in str(errors[0])


def test_a_failing_host_body_on_one_rank_is_reported_without_hanging_its_peers(npu_dispatch):
    """The failing rank still joins the digest collective, so the others get a mismatch, not a wait."""
    world = FakeWorld(4)

    def boom(x, hash_ids, token_mask):
        raise MemoryError("simulated host allocation failure")

    def rank_body(rank: int):
        gather, broadcast = world.collectives(rank)
        state = HostEngramState(rank, world.world, gather=gather, broadcast=broadcast)
        x, ids, mask = inputs()
        body = make_body() if rank != 2 else boom
        return rank0_host_engram(counting(body, Counter(), rank), x, ids, mask, state)

    _, errors = run_ranks(world, rank_body)
    assert set(errors) == set(range(4))
    assert isinstance(errors[2], MemoryError)
    assert all(isinstance(errors[rank], RuntimeError) for rank in (0, 1, 3))
    assert "'result'" in str(errors[0])


def test_the_debug_flag_re_proves_on_every_call(npu_dispatch):
    world = FakeWorld(4)
    results, errors, states, computes = one_call_per_rank(
        world, state_kwargs={"verify_always": True}, calls=2
    )
    assert not errors, errors
    # Both calls compute on every rank; rank 0 reuses its verified result instead of computing again.
    assert computes == Counter({0: 2, 1: 2, 2: 2, 3: 2})
    assert all(state.verified_calls == 2 for state in states.values())
    assert world.counters["gather"] == world.world * 5 * 2
    assert torch.equal(bits(results[1][1]), bits(results[0][0]))


def test_the_debug_flag_defaults_off_and_is_read_from_the_environment(monkeypatch):
    monkeypatch.delenv(ENGRAM_RANK0_ENV, raising=False)
    monkeypatch.delenv(ENGRAM_VERIFY_ALWAYS_ENV, raising=False)
    assert enabled_from_environment() is False
    assert verify_always_from_environment() is False
    for spelling in ("1", "true", "TRUE", "on", "yes", " 1 "):
        monkeypatch.setenv(ENGRAM_RANK0_ENV, spelling)
        monkeypatch.setenv(ENGRAM_VERIFY_ALWAYS_ENV, spelling)
        assert enabled_from_environment() is True
        assert verify_always_from_environment() is True
    for spelling in ("0", "", "no", "off", "false"):
        monkeypatch.setenv(ENGRAM_RANK0_ENV, spelling)
        monkeypatch.setenv(ENGRAM_VERIFY_ALWAYS_ENV, spelling)
        assert enabled_from_environment() is False
        assert verify_always_from_environment() is False


# ------------------------------------------------------------------------------------------------
# Bytes, not arithmetic
# ------------------------------------------------------------------------------------------------


def test_signed_zero_and_subnormal_bits_cross_the_wire_exactly(npu_dispatch):
    """The wire's whole point: -0.0 and the smallest BF16 subnormal arrive unchanged."""

    def hostile(x, hash_ids, token_mask):
        out = torch.zeros(x.shape, dtype=torch.bfloat16)
        out.view(torch.int16)[..., 0] = -32768   # 0x8000: negative zero
        out.view(torch.int16)[..., 1] = 1        # 0x0001: smallest positive subnormal
        out.view(torch.int16)[..., 2] = 0x7F80   # +inf, also a bit pattern
        return out

    expected = hostile(torch.zeros(1, 1, HC_MULT, DIM, dtype=torch.bfloat16), None, None)
    for wire in ("int8", "bf16"):
        world = FakeWorld(4)
        results, errors, states, _ = one_call_per_rank(
            world, body_factory=lambda rank: hostile, state_kwargs={"wire": wire}
        )
        assert not errors, errors
        for rank in range(4):
            assert torch.equal(bits(results[rank][0]), bits(expected)), f"{wire} wire, rank {rank}"
        assert states[0].wire == wire


def test_the_bf16_wire_moves_elements_and_the_int8_wire_moves_bytes(npu_dispatch):
    elements = 1 * 1 * HC_MULT * DIM
    for wire, per_result in (("int8", elements * 2), ("bf16", elements)):
        world = FakeWorld(2)
        _, errors, _, _ = one_call_per_rank(world, state_kwargs={"wire": wire}, calls=1)
        assert not errors, errors
        assert world.bytes["broadcast"] == world.world * per_result


def test_an_unknown_wire_is_refused():
    with pytest.raises(ValueError, match="wire must be 'int8' or 'bf16'"):
        HostEngramState(0, 1, wire="fp32")


def test_the_collective_device_is_the_current_card_or_the_host():
    """Resolution only; the helper never selects a device, so this reads without setting."""
    resolved = host_engram._collective_device()
    if hasattr(torch, "npu") and torch.npu.is_available():
        assert resolved.type == "npu"
    else:
        assert resolved == torch.device("cpu")


def test_from_distributed_plumbs_the_real_collective_calls(monkeypatch, npu_dispatch):
    """The closures `from_distributed` builds carry int8 payloads to the group and back."""
    import sys
    import types

    real = sys.modules["torch.distributed"]
    gathered: list[int] = []

    def all_gather(parts, local):
        gathered.append(int(local.numel()))
        assert local.dtype == torch.int8, "the proof wire is int8, never floating arithmetic"
        for part in parts:
            part.copy_(local)

    def broadcast(tensor, src=0):
        assert src == 0
        assert tensor.dtype == torch.int8

    fake = types.SimpleNamespace(
        is_available=lambda: True, is_initialized=lambda: True,
        get_world_size=lambda: 2, get_rank=lambda: 0, get_backend=lambda: "hccl",
        all_gather=all_gather, broadcast=broadcast,
    )
    monkeypatch.setattr(sys, "modules", {**sys.modules, "torch.distributed": fake})
    monkeypatch.setattr(torch, "distributed", fake, raising=False)
    try:
        state = HostEngramState.from_distributed(device="cpu")
        assert (state.rank, state.world, state.backend) == (0, 2, "hccl")
        computes: Counter = Counter()
        x, ids, mask = inputs()
        result = rank0_host_engram(counting(make_body(), computes, 0), x, ids, mask, state)
    finally:
        monkeypatch.undo()
        sys.modules["torch.distributed"] = real
    assert state.verified is True
    assert torch.equal(bits(result), bits(make_body()(x, ids, mask)))
    # One digest collective, then the payloads that fit the raw bound, all as int8 vectors:
    # 4 digests, the BF16 span, int64 hash ids, the one-byte-per-token bool mask, the result.
    assert gathered == [4 * 32, 2 * 1 * 1 * HC_MULT * DIM, 1 * 1 * COLS * 8, 1 * 1,
                        2 * 1 * 1 * HC_MULT * DIM]
    assert state.broadcasts == 1


# ------------------------------------------------------------------------------------------------
# Validation and isolation
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("rank", "world"), [(-1, 1), (2, 2), (0, 0)])
def test_an_impossible_rank_or_world_is_refused(rank, world):
    with pytest.raises(ValueError):
        HostEngramState(rank, world)


def test_a_world_greater_than_one_without_collectives_is_refused():
    with pytest.raises(ValueError, match="injected `gather` and `broadcast`"):
        HostEngramState(0, 2)


def test_from_distributed_refuses_an_uninitialized_process_group():
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        pytest.skip("this process already has a group; the refusal cannot be observed")
    with pytest.raises(RuntimeError, match="initialized torch.distributed process group"):
        HostEngramState.from_distributed()


def test_two_engram_modules_do_not_share_a_proof(npu_dispatch):
    """One state per module: the second module proves its own first call."""
    world = FakeWorld(2)
    computes: Counter = Counter()

    def rank_body(rank: int):
        gather, broadcast = world.collectives(rank)
        first = HostEngramState(rank, 2, gather=gather, broadcast=broadcast)
        second = HostEngramState(rank, 2, gather=gather, broadcast=broadcast)
        assert not second.verified, "a fresh module must not inherit a proof"
        x, ids, mask = inputs()
        produced = []
        for state in (first, second):
            produced.append(rank0_host_engram(counting(make_body(), computes, rank), x, ids, mask,
                                              state))
        return produced, first, second

    results, errors = run_ranks(world, rank_body)
    assert not errors, errors
    for rank in range(2):
        produced, first, second = results[rank]
        assert first.verified and second.verified
        assert first.verified_calls == 1 and second.verified_calls == 1
        assert torch.equal(bits(produced[0]), bits(produced[1]))
    # Two modules, two proofs, then two rank-0 broadcasts.
    assert world.counters["gather"] == 2 * 5 * 2
    assert computes == Counter({0: 2, 1: 2})


def test_no_rank_changes_the_process_device(monkeypatch, npu_dispatch):
    """The helper never selects a device: the caller's current device is the one used."""
    recorder: list[object] = []
    fake_npu = type("FakeNpu", (), {"set_device": staticmethod(lambda *args: recorder.append(args))})()
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)

    world = FakeWorld(2)
    _, errors, _, _ = one_call_per_rank(world, calls=2)
    assert not errors, errors
    assert recorder == [], "the host Engram must not set a process-wide device"


def test_a_prefill_span_and_then_a_one_token_decode_span_both_work(npu_dispatch):
    """The real call pattern: prefill verifies on `tokens` positions, decode then sends one.

    The buffer is sized per call from the activation, so the proof does not pin the span. This is
    the case that would first break a real run if it did: the very first generated token arrives as
    a one-token span after a longer prompt.
    """
    world = FakeWorld(2)
    computes: Counter = Counter()
    states: dict[int, HostEngramState] = {}

    def rank_body(rank: int):
        gather, broadcast = world.collectives(rank)
        state = HostEngramState(rank, 2, gather=gather, broadcast=broadcast)
        states[rank] = state
        body = counting(make_body(), computes, rank)
        prompt_x, prompt_ids, prompt_mask = inputs(tokens=4)
        first = rank0_host_engram(body, prompt_x, prompt_ids, prompt_mask, state)
        step_x, step_ids, step_mask = inputs(tokens=1, seed=9)
        second = rank0_host_engram(body, step_x, step_ids, step_mask, state)
        return first, second, prompt_x, prompt_ids, prompt_mask, step_x, step_ids, step_mask

    results, errors = run_ranks(world, rank_body)
    assert not errors, errors
    reference = make_body()
    for rank in range(2):
        first, second, px, pids, pmask, sx, sids, smask = results[rank]
        assert torch.equal(bits(first), bits(reference(px, pids, pmask)))
        assert torch.equal(bits(second), bits(reference(sx, sids, smask)))
        assert first.shape == px.shape and second.shape == sx.shape
    assert states[0].verified_calls == 1, "the decode span reuses the prefill's proof"
    # The proof computes once per rank on the prompt span; the decode span computes on rank 0 only
    # (its verified result is reused, not recomputed -- that is the whole saving).
    assert computes == Counter({0: 2, 1: 1})
    # Each broadcast carries its own call's result size: 4 tokens then 1 token.
    assert world.bytes["broadcast"] == 2 * (2 * (4 * HC_MULT * DIM) + 2 * (1 * HC_MULT * DIM))


# ------------------------------------------------------------------------------------------------
# The probe end to end, on one rank
# ------------------------------------------------------------------------------------------------


def test_the_probe_runs_end_to_end_on_a_single_rank_group(tmp_path, capsys):
    """`ascend_host_engram.main` on a real (one-rank) process group, CPU only."""
    import torch.distributed as dist

    if not dist.is_available():
        pytest.skip("this torch build has no torch.distributed")
    store = tmp_path / "probe_store"
    dist.init_process_group(backend="gloo", init_method=f"file://{store}", world_size=1, rank=0)
    try:
        assert host_engram.main(["--iterations", "2", "--device", "cpu"]) == 0
    finally:
        dist.barrier()
        dist.destroy_process_group()
    output = capsys.readouterr().out
    assert "PROBE" in output
    assert "'result_bits_identical_to_local_host_compute': True" in output
    # A one-rank group is the pass-through path: no proof is claimed and no collective is invented.
    assert "'world': 1" in output
    assert "'verified': False" in output
    assert "'gathers': 0" in output and "'broadcasts': 0" in output


# ------------------------------------------------------------------------------------------------
# The device the lead leases
# ------------------------------------------------------------------------------------------------


@pytest.mark.skipif(not os.environ.get(DEVICE_ENV),
                    reason=f"set {DEVICE_ENV}=npu:0 on a leased card to run the device case")
def test_the_byte_alias_is_metadata_on_the_device():
    """The wire primitive on a real card: an int8 alias of a BF16 result keeps every bit.

    This is the piece the portable tests stand in for with host tensors; it is not a claim that HCCL
    moved int8 across ranks -- that is what `ascend_host_engram.main` checks on eight ranks.
    """
    torch_npu = pytest.importorskip("torch_npu")
    del torch_npu
    device = torch.device(os.environ[DEVICE_ENV])
    host = torch.tensor([[0.0, -0.0, 2.0**-133, float("inf")]], dtype=torch.bfloat16)
    on_device = host.to(device)
    alias = on_device.view(torch.int8)
    assert alias.device == on_device.device and alias.dtype == torch.int8
    assert int(alias.numel()) == 2 * int(on_device.numel())
    back = alias.view(torch.bfloat16).view(host.shape)
    assert torch.equal(bits(back.cpu()), bits(host))
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize(device)