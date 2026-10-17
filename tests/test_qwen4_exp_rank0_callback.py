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
import socket
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
# the loop, on two real ranks
# -------------------------------------------------------------------------------------------


def _run_ranks(mode: str, world: int = 2, timeout: float = 120.0) -> list[dict]:
    """Run this file once per rank as a subprocess under one gloo group, and collect the results.

    A rank is a **process**, not a thread. `init_process_group` binds the process to a single group
    and a second call in the same interpreter is not a second rank -- the threads version of this
    test hung in `TCPStore` rather than exercising anything. So each rank is re-executed as
    `python <this file> --rank N --world W --mode M`, writes one JSON line to stdout, and the
    parent asserts on what came back.

    The group is what makes the test mean something: the step predicate is a real `dist.broadcast`
    of rank 0's decision, exactly the shape `ShardedWorkerMixin._step_sync` gives this runtime at
    TP>1. A rank 0 that leaves the loop early never broadcasts again and the others block in it, so
    the parent's timeout is the failure the fix removes.

    Every subprocess is killed on the way out, including on failure, so a hung rank cannot leak
    into the rest of the suite.
    """
    import json
    import subprocess

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    procs = []
    for rank in range(world):
        procs.append(
            subprocess.Popen(
                [
                    sys.executable,
                    str(pathlib.Path(__file__).resolve()),
                    "--rank", str(rank),
                    "--world", str(world),
                    "--mode", mode,
                    "--port", str(port),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )

    results: list[dict] = []
    try:
        for rank, proc in enumerate(procs):
            try:
                out, err = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                raise AssertionError(
                    f"rank {rank} never finished: its peers are blocked in a collective the "
                    f"departed rank will not enter (this is the hang the fix removes). "
                    f"stderr tail: {(proc.stderr.read() or '')[-600:]}"
                ) from None
            assert proc.returncode == 0, f"rank {rank} exited {proc.returncode}: {err[-800:]}"
            # The result line is the last JSON object on stdout; anything the runtime prints goes
            # before it and is ignored rather than parsed.
            line = [ln for ln in out.splitlines() if ln.startswith("{")][-1]
            results.append(json.loads(line))
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)

    return results


def _rank_body(rank: int, world: int, mode: str, port: int) -> dict:
    """One rank's share of the loop, run in its own process.

    `mode` picks the scenario: `raise` is the regression (rank 0's callback throws), `healthy` is
    the control that must still agree when nothing throws.
    """
    import torch.distributed as dist

    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world,
    )

    backend = _backend_without_loading()
    backend._rank = rank
    backend._world = world
    backend._distributed = True

    flag = torch.zeros(1, dtype=torch.int32, device="cpu")

    def on_step() -> bool:
        # Rank 0's local decision, broadcast: the one collective the loop has per step, and the
        # same seam `_step_sync` provides. A rank that left without taking part strands its peers.
        if rank == 0:
            flag.fill_(1 if backend._callback_failure else 0)
        dist.broadcast(flag, src=0)
        return bool(int(flag.item()))

    if mode == "raise":

        def boom(token: int) -> None:
            raise RuntimeError("streamer failed")

        user_callback = boom if rank == 0 else None
    else:
        user_callback = (lambda token: None) if rank == 0 else None

    produced, stats = runtime.generate(
        _ScriptedModel([3, 4, 5, 6, 7, 8]),
        runtime.TPContext(rank, world, torch.device("cpu"), True),
        torch.tensor([[1, 2]], dtype=torch.long),
        max_new_tokens=8,
        chunk_size=8,
        eos_token_ids=(),
        on_token=backend._rank0_callback(user_callback, "req-tp"),
        on_step=on_step,
        local_eos=[False],
    )

    dist.destroy_process_group()
    return {
        "rank": rank,
        "produced": list(produced),
        "stopped": stats["stopped"],
        "failure": [str(e) for e in backend._callback_failure],
    }


def test_a_raising_rank0_callback_does_not_strand_the_other_ranks():
    """The regression: rank 0's callback raises, and every rank still leaves the loop.

    On the broken code rank 0 unwinds at the callback and never reaches another broadcast, so the
    other rank sits in `dist.broadcast` until `_run_ranks` times out.
    """
    results = _run_ranks("raise")

    for result in results:
        assert result["stopped"] == "cancel", (
            f"rank {result['rank']} ended as {result['stopped']!r}, not through the agreed "
            f"cancellation"
        )
        # Rank 0's failure is acted on at the next step boundary, so at most the one token already
        # in flight is produced.
        assert len(result["produced"]) <= 1

    rank0 = [r for r in results if r["rank"] == 0][0]
    assert rank0["failure"] == ["streamer failed"], "rank 0 did not record the callback's failure"


def test_a_healthy_run_still_agrees_without_a_callback_failure():
    """The same harness with no raise, so the test above measures the raise and not the shape."""
    results = _run_ranks("healthy")

    for result in results:
        assert result["stopped"] == "length", f"rank {result['rank']} stopped early"
        # Six scripted tokens and an eight-token budget: the model repeats its last one once the
        # script runs out, so the answer is eight long and both ranks must agree on all of it.
        assert result["produced"] == [3, 4, 5, 6, 7, 8, 8, 8]
        assert result["failure"] == []


def _main(argv: list[str]) -> int:
    """The per-rank entry point `_run_ranks` re-executes this file with."""
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world", type=int, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args(argv)

    result = _rank_body(args.rank, args.world, args.mode, args.port)
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
