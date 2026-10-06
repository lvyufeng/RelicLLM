"""The sharded-worker protocol: what every rank-mirrored runtime does the same way.

MiMo and Qwen4-Exp are both served as one process a rank over a process group, and both reach
their loops through the same six steps: a rank announces itself only after the load's barrier, rank
0 broadcasts one payload a request before it enters a generation its peers have to be inside, every
step asks for a stop flag every rank agrees on, a worker serves the payload alone, a worker barriers
before the group is torn down, and rank 0 broadcasts a shutdown before the base reaps the group.

The six were written out twice, byte for byte, and the reason is not hard to state: nothing in
:class:`~relicllm.backends.runtime_engine.RankedWorker` said they were one protocol, so each runtime
carried its own copy of the reasoning and a change to any invariant was a change in two places that
had to agree. This mixin is that one place.

It is a mixin rather than a base because a sharded runtime is a :class:`RuntimeAdapter` -- the
request lifecycle, the tokenizer, the result shape -- *and* a sharded worker. A subclass puts it
first (``class MimoBackend(ShardedWorkerMixin, RuntimeAdapter)``) so these methods win over
``RankedWorker``'s defaults and :meth:`close` still reaches ``BackendBase`` through the MRO.

What is *not* here is what genuinely differs per runtime and stays on the adapter: the loop itself
(``_loop``), the load, the distributed setup, the payload runner (``_run_payload``), and -- when a
runtime's payload is not just the prompt and the budget -- the payload builder, which a subclass
overrides in :meth:`_worker_payload`.

V4.1 is deliberately not in this family. It reaches the same six steps, but two of them go through a
doorbell rather than a plain collective -- its rank 0 broadcasts through :meth:`_broadcast` and its
idle workers wait on the host rather than inside an NCCL poll kernel -- so pulling it in would be a
second branch in each method rather than a second caller of one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from relicllm.api import GenerationRequest

from .runtime_engine import RankedWorker


class ShardedWorkerMixin:
    """The rank-mirrored serving protocol MiMo and Qwen4-Exp share.

    Every method here reads only ``self._world``, ``self._rank``, ``self._device``, ``self._closed``
    and ``self._ready`` -- the state :class:`RankedWorker` and :class:`BackendBase` already give a
    subclass -- plus :meth:`_is_cancelled` and :meth:`_run_payload`, which the adapter owns.
    """

    def _publish_ready(self) -> None:
        """Barrier across the ranks, then mark this rank ready.

        The barrier comes first because a rank that is ready before its peers have loaded would be
        answered by a parent that thinks every rank is up, and the mismatch would surface later
        inside a collective rather than here, where it can be explained.
        """
        if self._world > 1:
            import torch.distributed as dist

            dist.barrier()
        self._ready = True

    def _step_sync(
        self, request_id: str, local: Callable[[], bool] | None = None
    ) -> Callable[[], bool]:
        """A per-step "should this loop stop", agreed on by every rank.

        Rank 0 reads its own cancel flag and broadcasts it; every other rank reads what it was sent.
        That message is what makes a client's disconnect reach a loop whose every other line is
        deterministic and local -- and it is *this* collective rather than a local flag because a
        rank that left on its own would leave its peers inside a layer's all-reduce.

        ``local`` is a condition only rank 0 can evaluate -- the stream's stop-string check -- and it
        is folded into the same flag for exactly that reason. A stop string is found on one rank and
        nowhere else, so a rank 0 that unwound on its own to report it would leave the others in a
        layer; asking here instead makes the answer stop one step later, at a boundary every rank
        arrives at together.
        """
        if self._world <= 1:

            def alone() -> bool:
                return self._is_cancelled(request_id) or (local is not None and local())

            return alone

        import torch
        import torch.distributed as dist

        flag = torch.zeros(1, dtype=torch.int32, device=self._device)

        def agreed() -> bool:
            if self._rank == 0:
                stop = self._is_cancelled(request_id) or (local is not None and local())
                flag.fill_(1 if stop else 0)
            dist.broadcast(flag, src=0)
            return bool(int(flag.item()))

        return agreed

    def _worker_payload(
        self, request: GenerationRequest, prompt_ids: Sequence[int], budget: int
    ) -> dict[str, Any]:
        """What rank 0 broadcasts for one request, as the worker loop will read it back.

        The prompt and the budget, and nothing else, because the workers reproduce the run rather
        than re-derive it: they are not given the prompt *text* -- tokenization is rank 0's, and a
        rank that tokenized it itself would be a second renderer that could disagree -- and a loop
        that samples greedily needs no sampler here. A runtime whose loop does read the sampler
        widens this by overriding it, adding the keys its own worker loop takes.
        """
        return {
            "op": "generate",
            "request_id": request.request_id,
            "prompt_ids": [int(token) for token in prompt_ids],
            "max_new_tokens": int(budget),
        }

    def _dispatch(
        self, request: GenerationRequest, prompt_ids: Sequence[int], budget: int
    ) -> None:
        """Hand every other rank the request, so that the work below is mirrored and not soloed.

        **The collective is what makes this mandatory rather than tidy.** Every routed layer closes
        with an ``all_reduce``, so a rank that is not running the same request is not idle -- it is
        at a *different* collective, and NCCL answers a mismatch by hanging both sides. Rank 0
        therefore may not enter a generation the workers have not been told about, and the broadcast
        below is the only thing that tells them: the worker loop blocks on this exact call, one
        payload a request, and a request that never arrives is a rank 0 that deadlocks on its own
        first expert layer rather than a rank 0 that quietly returns a wrong answer.

        Only rank 0 sends. The payload's keys are the loop's arguments, so they are the subclass's
        to widen in :meth:`_worker_payload`; the send itself and the guards around it are not.
        """
        if self._world <= 1 or self._rank != 0:
            return
        import torch.distributed as dist

        if not dist.is_initialized():
            return
        payload = self._worker_payload(request, prompt_ids, budget)
        dist.broadcast_object_list([payload], src=0)

    def _run_worker_request(self, payload: Mapping[str, Any]) -> None:
        """Serve one request, which here takes the payload and nothing else.

        The base body is ``v41``'s call shape -- ``_run_payload(payload, request, on_token)`` --
        because that method's surface is the union of the serial path's and the worker loop's. This
        runtime's runs a payload alone: nothing here has a per-request hook a worker rank could
        pass, and the two arguments it does not take are not ones it should grow to ignore.
        """
        self._run_payload(payload)

    def _worker_drained(self) -> None:
        """Barrier before rank 0 tears the group down.

        A worker that reaches the end of the loop is done serving, but its last request's
        collectives may still be in flight on the device; the barrier is what makes "the loop has
        ended" on a worker mean "rank 0 may now destroy the group" rather than a race the NCCL
        teardown usually wins.
        """
        import torch.distributed as dist

        dist.barrier()

    def close(self) -> None:
        """Broadcast the shutdown to the workers, then close the base.

        The shutdown has to be sent before the base reaps the group: once the ranks this process
        spawned are gone there is nobody left to receive it, and once this rank leaves the group a
        broadcast is a collective without a peer. A worker rank that reached the end of
        :meth:`~relicllm.backends.runtime_engine.RankedWorker.run_worker` is already out of the
        message it was sent, so it has nothing to send and this is rank 0's alone.
        """
        already_closed = self._closed
        if not already_closed and self._world > 1 and self._rank == 0:
            try:
                import torch.distributed as dist

                dist.broadcast_object_list([{"op": RankedWorker._WORKER_SHUTDOWN}], src=0)
            except Exception:
                # A peer that already left is not this rank's problem, and close() must not raise.
                pass
        super().close()


__all__ = ["ShardedWorkerMixin"]