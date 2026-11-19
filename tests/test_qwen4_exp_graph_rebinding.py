"""Graph bindings suspend in reverse stack order and resume without new allocations."""
from types import SimpleNamespace
import pytest
import torch
from relicllm.models.qwen4_exp.verify_graph import VerifyGraph, DraftGraph, _MISSING


def owner(cls,target,value,table):
    graph=cls.__new__(cls)
    graph._saved_attrs=[(target,"staged",target.__dict__.get("staged",_MISSING))]
    graph._saved_rows_buffers=[(table,table._rows_buffer)]
    graph._bindings_retired=False
    graph._released=False
    target.staged=value
    table._rows_buffer=value
    return graph


def test_nested_request_bindings_suspend_resume_and_keep_buffers():
    target=SimpleNamespace();original=torch.zeros(3);table=SimpleNamespace(_rows_buffer=original)
    tail=owner(VerifyGraph,target,torch.ones(3),table)
    wide=owner(VerifyGraph,target,torch.full((3,),2),table)
    tail_buffer=tail._saved_rows_buffers[0][1]
    wide_buffer=table._rows_buffer
    wide.suspend_bindings();tail.suspend_bindings()
    assert not hasattr(target,"staged") and table._rows_buffer is original
    tail.resume_bindings();wide.resume_bindings()
    assert target.staged is wide_buffer and table._rows_buffer is wide_buffer
    assert tail._saved_rows_buffers[0][1] is tail_buffer
    assert not wide._bindings_retired and not tail._bindings_retired
    wide.suspend_bindings();tail.suspend_bindings()
    # Retiring an already suspended bundle must not resurrect the outer binding.
    wide.retire_bindings();tail.retire_bindings()
    assert not hasattr(target,"staged") and table._rows_buffer is original
    assert wide._bindings_retired and tail._bindings_retired


@pytest.mark.parametrize("cls", [VerifyGraph,DraftGraph])
def test_suspension_is_idempotent_and_retired_owner_cannot_resume(cls):
    target=SimpleNamespace();table=SimpleNamespace(_rows_buffer=None)
    graph=owner(cls,target,torch.ones(2),table)
    graph.suspend_bindings();graph.suspend_bindings()
    assert not hasattr(target,"staged")
    graph.resume_bindings();graph.resume_bindings()
    assert hasattr(target,"staged")
    graph.suspend_bindings();graph.retire_bindings()
    with pytest.raises(RuntimeError,match="retired"):
        graph.resume_bindings()


@pytest.mark.parametrize("cls", [VerifyGraph,DraftGraph])
def test_suspended_recording_cannot_replay(cls):
    graph=cls.__new__(cls);graph._bindings_suspended=True;graph._bindings_retired=False
    with pytest.raises(RuntimeError,match="temporarily suspended"):
        graph.replay(0,[1] if cls is VerifyGraph else 1)
