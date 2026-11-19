"""SSE backpressure must not prevent the TP rank-0 callback reaching cancellation."""
from types import SimpleNamespace
import queue
import threading
import pytest
from relicllm.api import RequestCancelledError,TokenEvent
from relicllm.backends.qwen4_exp_backend import _MTPEventQueue
from relicllm.backends import base


def adapter():
    state=SimpleNamespace(_mtp_closing=False,cancelled=False)
    state._is_cancelled=lambda _id:state.cancelled
    return state


def test_a_closed_reader_unblocks_a_backpressured_producer():
    state=adapter();events=_MTPEventQueue(state,'request',1)
    events.put(TokenEvent(request_id='request',token_id=1))
    started=threading.Event();errors=[]
    def produce():
        started.set()
        try:events.put(TokenEvent(request_id='request',token_id=2))
        except BaseException as exc:errors.append(exc)
    thread=threading.Thread(target=produce);thread.start();assert started.wait(1)
    events.close_reader();thread.join(1)
    assert not thread.is_alive() and len(errors)==1 and isinstance(errors[0],RequestCancelledError)
    events.put(None)
    assert events.get_nowait() is None


def test_api_cancellation_keeps_a_terminal_event_deliverable_when_full():
    state=adapter();events=_MTPEventQueue(state,'request',1)
    events.put(TokenEvent(request_id='request',token_id=1));state.cancelled=True
    end=TokenEvent(request_id='request',finish_reason='cancelled')
    events.put(end)
    assert events.get_nowait() is end
    events.put(None);assert events.get_nowait() is None


def test_uncancelled_queue_preserves_standard_nonblocking_and_timeout_semantics():
    state=adapter();events=_MTPEventQueue(state,'request',1)
    events.put(1)
    with pytest.raises(queue.Full):events.put(2,block=False)
    with pytest.raises(queue.Full):events.put(2,timeout=0)
    assert events.get_nowait()==1


def test_stream_generator_close_cancels_before_joining_the_producer(monkeypatch):
    monkeypatch.setattr(base,'_STREAM_QUEUE_DEPTH',1)
    state=adapter();state.cleared=False
    def cancel(_id):state.cancelled=True;return True
    state.cancel=cancel
    state._clear_request=lambda _id:setattr(state,'cleared',True)
    state._stream_event_queue=lambda _id,depth:_MTPEventQueue(state,_id,depth)
    def produce(request,events,_outcome):
        for token in range(100):events.put(TokenEvent(request_id=request.request_id,token_id=token))
    state._produce=produce
    run=base._StreamRun(state,SimpleNamespace(request_id='request'),'mtp-test')
    run.start();iterator=run.drain();assert next(iterator).token_id==0
    iterator.close()
    assert not run.thread.is_alive() and state.cancelled and state.cleared
    assert isinstance(run.outcome['error'],RequestCancelledError)
