"""A rank-0 callback that raises has to stop the group, not just rank 0.

`generate` calls `on_token` under `ctx.rank == 0` alone. Every other rank has already moved on to
the next step's collective by the time that call returns, so an exception escaping it unwinds rank 0
out of a broadcast its peers are blocked in -- and the request hangs instead of failing:

    WorkNCCL(OpType=BROADCAST, NumelIn=1, NumelOut=1, Timeout(ms)=600000)
    ran for 600000 milliseconds before timing out.

This is the *second* rank-0-only exit in this loop and it is independent of the end-of-sequence one
`test_qwen4_exp_eos_collective.py` covers: that one is a token disagreement, this one is a single
rank raising where the others cannot see it. Measured on 8 x RTX 4090 by injecting a raise at the
callback: rank 0 sat at 0% utilization while the other seven spun at 100% until the watchdog fired,
and a second request to the still-`ready` server never came back.

The contract these pin is `Qwen4ExpBackend._rank0_callback`: a raise out of the user's callback must
become a cancellation that rides the broadcast every rank already reaches, and must surface as an
exception from the request rather than as a hang.
"""

from __future__ import annotations

import pathlib
import sys
import threading

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest
import torch

from relicllm.backends import qwen4_exp_backend
from relicllm.models.qwen4_exp import runtime


class _ScriptedModel:
    """A model whose next token is dictated, so the loop runs without a checkpoint.

    Copied in spirit from `test_qwen4_exp_eos_collective.py`: `generate` only asks for `make_cache`,
    `forward` and `profiler`, and reads the token off `gather_logits(...).argmax(...)`. A one-hot row
    makes that argmax exact, so the script is the token stream.
    """

    def __init__(self, tokens: list[int], vocab: int = 16) -> None:
        self._tokens = list(tokens)
        self._index = 0
        self._vocab = vocab
        self.profiler = None

    def make_cache(self, *, batch_size: int, max_seq_len: int):
        return object()

    def forward(self, ids, *, cache=None, past_len=0, last_token_only=False):
        row = torch.zeros(ids.shape[0], ids.shape[1], self._vocab)
        token = self._tokens[min(self._index, len(self._tokens) - 1)]
        self._index += 1
        row[:, -1, token] = 10.0
        return row


def _backend_without_loading() -> qwen4_exp_backend.Qwen4ExpBackend:
    """A backend instance with a world of one, built without touching a checkpoint.

    `_rank0_callback` reads `self._callback_failure` and calls `self.cancel`, neither of which needs
    a loaded model, so the adapter is constructed the way the routing tests construct it and no
    `load` is attempted.
    """
    from relicllm.api import EngineArgs

    return qwen4_exp_backend.Qwen4ExpBackend(EngineArgs(model="/nonexistent", backend="qwen4_exp"))


# -------------------------------------------------------------------------------------------
# the wrapper's own contract
# -------------------------------------------------------------------------------------------


def test_a_raising_callback_is_recorded_and_not_propagated():
    """The wrapper must not let the raise out: propagating is the hang, by construction."""
    backend = _backend_without_loading()
    calls: list[int] = []

    def boom(token: int) -> None:
        calls.append(token)
        raise RuntimeError("streamer failed")

    guarded = backend._rank0_callback(boom, "req-1")
    assert guarded is not None

    guarded(7)  # must not raise

    assert calls == [7], "the callback did not run"
    assert len(backend._callback_failure) == 1
    assert isinstance(backend._callback_failure[0], RuntimeError)


def test_the_recorded_failure_is_what_the_request_raises():
    """`_raise_callback_failure` is the frame that owns the request; it re-raises, once."""
    backend = _backend_without_loading()
    backend._callback_failure.append(RuntimeError("streamer failed"))

    with pytest.raises(RuntimeError, match="streamer failed"):
        backend._raise_callback_failure()

    assert backend._callback_failure == [], "the failure was not popped"
    backend._raise_callback_failure()  # a clean request must not raise


def test_a_failing_callback_is_not_called_again():
    """After the first raise the loop is only running out the steps its peers still owe."""
    backend = _backend_without_loading()
    calls: list[int] = []

    def boom(token: int) -> None:
        calls.append(token)
        raise RuntimeError("streamer failed")

    guarded = backend._rank0_callback(boom, "req-2")
    assert guarded is not None

    guarded(1)
    guarded(2)
    guarded(3)

    assert calls == [1], f"the callback ran {len(calls)} times after failing"


def test_a_healthy_callback_is_forwarded_untouched():
    """The wrapper is not allowed to change a run that has no failure in it."""
    backend = _backend_without_loading()
    seen: list[int] = []

    guarded = backend._rank0_callback(seen.append, "req-3")
    assert guarded is not None
    guarded(11)
    guarded(12)

    assert seen == [11, 12]
    assert backend._callback_failure == []


def test_no_callback_stays_no_callback():
    """A serial request passes none, and the wrapper must not invent one."""
    backend = _backend_without_loading()
    assert backend._rank0_callback(None, "req-4") is None


# -------------------------------------------------------------------------------------------
# the loop, under a real two-rank process group
# -------------------------------------------------------------------------------------------


def _run_two_ranks(body, *, timeout: float = 30.0) -> list:
    """Run `body(rank, world)` on two threads under a gloo group, and fail if it does not finish.

    The point of the group is the broadcast in the step predicate: with it, a rank that leaves the
    loop early leaves the other in `dist.broadcast` forever. Without it this test would pass on the
    broken code, because there would be nothing to block in.

    `body` is called on both threads and its exception, if any, is re-raised here.
    """
    import torch.distributed as dist

    store = dist.TCPStore("127.0.0.1", 0, 2, True, timeout=timeout)
    port = store.port
    results: list = [None, None]
    errors: list[BaseException | None] = [None, None]
    barrier = threading.Barrier(2)

    def worker(rank: int) -> None:
        try:
            dist.init_process_group(
                backend="gloo",
                init_method=f"tcp://127.0.0.1:{port}",
                rank=rank,
                world_size=2,
            )
            barrier.wait(timeout=timeout)
            results[rank] = body(rank, 2)
        except BaseException as exc:  # noqa: BLE001 - reported through `errors`
            errors[rank] = exc
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()

    threads = [threading.Thread(target=worker, args=(rank,)) for rank in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout)

    assert not any(thread.is_alive() for thread in threads), (
        "a rank never left the loop: the peers are blocked in a collective the departed rank "
        "will not enter (this is the hang the fix removes)"
    )

    for error in errors:
        if error is not None:
            raise error
    return results


def _ctx_for(rank: int, world: int):
    return runtime.TPContext(rank, world, torch.device("cpu"), True)


def test_a_raising_rank0_callback_does_not_strand_the_other_ranks():
    """The regression: rank 0's callback raises, and every rank still leaves the loop.

    The predicate is a real `dist.broadcast` of rank 0's decision, which is the shape
    `ShardedWorkerMixin._step_sync` gives this runtime at TP>1. A rank 0 that unwound on the raise
    would never broadcast again, so the other rank would sit in it until the join timeout above.
    """
    import torch.distributed as dist

    def body(rank: int, world: int):
        backend = _backend_without_loading()
        backend._rank = rank
        backend._world = world
        backend._distributed = True

        flag = torch.zeros(1, dtype=torch.int32, device="cpu")

        def on_step() -> bool:
            # Rank 0's local decision, broadcast -- the one collective the loop has per step.
            if rank == 0:
                flag.fill_(1 if backend._callback_failure else 0)
            dist.broadcast(flag, src=0)
            return bool(int(flag.item()))

        def boom(token: int) -> None:
            raise RuntimeError("streamer failed")

        guarded = backend._rank0_callback(boom if rank == 0 else None, "req-tp")

        return runtime.generate(
            _ScriptedModel([3, 4, 5, 6, 7, 8]),
            _ctx_for(rank, world),
            torch.tensor([[1, 2]], dtype=torch.long),
            max_new_tokens=8,
            chunk_size=8,
            eos_token_ids=(),
            on_token=guarded,
            on_step=on_step,
            local_eos=[False],
        )

    results = _run_two_ranks(body)

    for produced, stats in results:
        assert stats["stopped"] == "cancel", (
            f"the loop ended as {stats['stopped']!r}, not through the agreed cancellation"
        )
        assert len(produced) <= 1, "rank 0's failure was not acted on until the next step boundary"


def test_a_healthy_run_still_agrees_without_a_callback_failure():
    """The same harness with no raise, so the test above is measuring the raise and not the shape."""
    import torch.distributed as dist

    def body(rank: int, world: int):
        backend = _backend_without_loading()
        backend._rank = rank
        backend._world = world
        backend._distributed = True

        stop = [False]

        def on_step() -> bool:
            return stop[0]

        return runtime.generate(
            _ScriptedModel([3, 4, 5, 6, 7, 8]),
            _ctx_for(rank, world),
            torch.tensor([[1, 2]], dtype=torch.long),
            max_new_tokens=8,
            chunk_size=8,
            eos_token_ids=(),
            on_token=backend._rank0_callback(rank == 0 and (lambda token: None) or None, "req-ok"),
            on_step=on_step,
            local_eos=[False],
        )

    results = _run_two_ranks(body)

    for produced, stats in results:
        assert stats["stopped"] == "length"
        assert produced == [3, 4, 5, 6, 7, 8]