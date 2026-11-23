"""CPU and metadata spies for the explicit B1 sparse-layout owner; no NPU executes."""
from types import SimpleNamespace

import pytest
import torch

from relicllm.models.deepseek_v4_1 import attention as attention_module
from relicllm.models.deepseek_v4_1 import ascend_sparse_attention_layout as binding


class Attention:
    def __init__(self, operands):
        self.operands = operands
        self.attn_sink = operands[2]
        self.action = None

    def forward(self, x, *args, **kwargs):
        if self.action is not None:
            return self.action(x)
        return attention_module.sparse_attn(*self.operands)


def operands(topk=65):
    q = torch.zeros((1, 1, 3, 512), dtype=torch.bfloat16)
    kv = torch.ones((1, 128, 512), dtype=torch.bfloat16)
    sink = torch.zeros(3)
    indices = torch.arange(topk, dtype=torch.int32).reshape(1, 1, topk)
    return q, kv, sink, indices, 512**-0.5


@pytest.fixture
def setup(monkeypatch):
    original_result, candidate_result = object(), object()
    original_calls, candidate_calls, owners = [], [], []

    def original(*args):
        original_calls.append(args)
        return original_result

    def operator(q, kv, indices, sink, scale, **kwargs):
        candidate_calls.append(((q, kv, sink, indices, scale), kwargs))
        return candidate_result

    monkeypatch.setattr(attention_module, "sparse_attn", original)

    def make(*, backend="torch", certified=False, budget=2**20):
        attention = Attention(operands())
        model = SimpleNamespace(layers=[SimpleNamespace(attn=attention)])
        owner = binding.SparseAttentionLayoutBinding(model, backend=backend,
            certified_valid_indices=certified, max_workspace_bytes=budget)
        monkeypatch.setattr(owner, "_resolve_operator", lambda: (operator, lambda topk, *, heads: 4096))
        owners.append(owner)
        return owner, attention, model

    result = SimpleNamespace(make=make, original=original, original_result=original_result,
        candidate_result=candidate_result, original_calls=original_calls, candidate_calls=candidate_calls,
        x=torch.ones((1, 1, 32), dtype=torch.bfloat16))
    yield result
    # Fixtures contain CPU tensors or inert fake-NPU metadata only. Restore even
    # when a failure intentionally leaves the owner registered for cleanup retry.
    for owner in reversed(owners):
        if owner.installed:
            owner.owner_thread = binding.get_ident()
            owner.current_attention = None
            attention_module.sparse_attn = owner.replacement_sparse
            binding._ACTIVE = owner
            for attention, replacement, *_ in owner.replacements:
                attention.forward = replacement
                setattr(attention, binding._OWNER, owner)
            monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=lambda device: None), raising=False)
            owner.close()
    assert binding._ACTIVE is None


def test_owned_b1_dispatch_preserves_argument_order_and_certificates(setup):
    owner, attention, _ = setup.make()
    owner.install()
    assert attention.forward(setup.x) is setup.candidate_result
    actual, kwargs = setup.candidate_calls[0]
    assert all(left is right for left, right in zip(actual, attention.operands))
    assert kwargs == {"backend": "torch", "certified_valid_indices": False, "max_workspace_bytes": 2**20}
    assert not setup.original_calls and owner.current_attention is None
    assert owner.describe()["selected_B1_calls"] == 1
    assert owner.describe()["workspace_budget_is_declared_tensor_estimate_not_allocator_or_graph_pool_peak"] is True
    owner.close()
    assert "forward" not in attention.__dict__ and not hasattr(attention, binding._OWNER)
    assert attention_module.sparse_attn is setup.original


def test_unowned_sparse_call_and_other_model_keep_the_original_provider(setup):
    owner, attention, _ = setup.make()
    other = Attention(attention.operands)
    owner.install()
    assert attention_module.sparse_attn(*attention.operands) is setup.original_result
    assert other.forward(setup.x) is setup.original_result
    assert not setup.candidate_calls and owner.describe()["outside_scope_bypasses"] == 2


@pytest.mark.parametrize("shape,dtype", [((1, 2, 32), torch.bfloat16), ((2, 1, 32), torch.bfloat16),
                                       ((1, 1, 32), torch.float32)])
def test_prefill_batch_or_input_dtype_uses_original_provider(setup, shape, dtype):
    owner, attention, _ = setup.make()
    owner.install()
    assert attention.forward(torch.zeros(shape, dtype=dtype)) is setup.original_result
    assert not setup.candidate_calls


@pytest.mark.parametrize("field", ["q_dim", "kv_limit", "index_dtype", "topk_limit", "sink_shape", "scale"])
def test_unsupported_sparse_metadata_falls_back_inside_owned_b1_scope(setup, field):
    owner, attention, _ = setup.make()
    q, kv, sink, indices, scale = attention.operands
    if field == "q_dim":
        q = torch.zeros((1, 1, 3, 256), dtype=torch.bfloat16)
    elif field == "kv_limit":
        kv = torch.zeros((1, 4097, 512), dtype=torch.bfloat16)
    elif field == "index_dtype":
        indices = indices.to(torch.int64)
    elif field == "topk_limit":
        indices = torch.zeros((1, 1, 513), dtype=torch.int32)
    elif field == "sink_shape":
        sink = torch.zeros(4)
    else:
        scale = float("nan")
    attention.operands = q, kv, sink, indices, scale
    owner.install()
    assert attention.forward(setup.x) is setup.original_result
    assert not setup.candidate_calls and owner.describe()["unsupported_bypasses"] == 1


def test_workspace_budget_falls_back_without_changing_metadata(setup):
    owner, attention, _ = setup.make(budget=4095)
    before = [value.clone() for value in attention.operands[:4]]
    owner.install()
    assert attention.forward(setup.x) is setup.original_result
    assert owner.describe()["budget_bypasses"] == 1 and not setup.candidate_calls
    assert all(torch.equal(value, expected) for value, expected in zip(attention.operands[:4], before))


def test_other_thread_never_enters_the_owned_layout_scope(setup, monkeypatch):
    owner, attention, _ = setup.make()
    owner.install()
    thread = owner.owner_thread
    with monkeypatch.context() as other:
        other.setattr(binding, "get_ident", lambda: thread + 1)
        assert attention.forward(setup.x) is setup.original_result
        with pytest.raises(RuntimeError, match="installation thread"):
            owner.close()
    assert not setup.candidate_calls and owner.installed


def test_nested_owned_forward_is_refused_and_scope_is_reset(setup):
    owner, attention, _ = setup.make()
    owner.install()
    attention.action = lambda x: attention.forward(x)
    with pytest.raises(RuntimeError, match="nested"):
        attention.forward(setup.x)
    assert owner.current_attention is None and not setup.candidate_calls


def test_operator_failure_does_not_leave_a_live_attention_scope(setup, monkeypatch):
    owner, attention, _ = setup.make()

    def fail(*args, **kwargs):
        raise RuntimeError("operator failed")

    monkeypatch.setattr(owner, "_resolve_operator", lambda: (fail, lambda topk, *, heads: 4096))
    owner.install()
    with pytest.raises(RuntimeError, match="operator failed"):
        attention.forward(setup.x)
    assert owner.current_attention is None


def test_second_owner_is_refused_before_altering_the_first(setup):
    owner, attention, _ = setup.make()
    second, _, _ = setup.make()
    owner.install()
    replacement = attention.forward
    with pytest.raises(RuntimeError, match="another model"):
        second.install()
    assert attention.forward is replacement and binding._ACTIVE is owner


def test_close_rejects_a_replaced_global_or_attention_owner(setup):
    owner, attention, _ = setup.make()
    owner.install()
    attention_module.sparse_attn = setup.original
    with pytest.raises(RuntimeError, match="sparse hook"):
        owner.close()
    attention_module.sparse_attn = owner.replacement_sparse
    replacement = attention.forward
    attention.forward = lambda *args: None
    with pytest.raises(RuntimeError, match="attention forward"):
        owner.close()
    attention.forward = replacement
    assert owner.installed


def test_existing_instance_forward_override_is_restored_exactly(setup):
    owner, attention, _ = setup.make()
    original = attention.forward
    attention.forward = original
    owner.install()
    owner.close()
    assert attention.__dict__["forward"] is original


@pytest.mark.parametrize("certified", [False, 0, 1, "yes", None])
def test_npu_requires_a_real_bounds_certificate_before_install(setup, certified):
    with pytest.raises(ValueError, match="certification"):
        setup.make(backend="triton", certified=certified)
    assert binding._ACTIVE is None


def test_npu_failed_fence_keeps_every_registered_reference_for_retry(setup, monkeypatch):
    owner, attention, _ = setup.make(backend="triton", certified=True)
    device = SimpleNamespace(type="npu", index=0)
    attention.attn_sink = SimpleNamespace(device=device)
    operator, budget = object(), object()
    owner._operator, owner._workspace_bytes = operator, budget
    owner.install()
    replacement = attention.forward

    def fail(value):
        assert value is device
        raise RuntimeError("fence failed")

    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=fail), raising=False)
    with pytest.raises(RuntimeError, match="fence failed"):
        owner.close()
    assert owner.installed and not owner.closed and binding._ACTIVE is owner
    assert owner.replacements and owner.devices and attention.forward is replacement
    assert owner._operator is operator and owner._workspace_bytes is budget
    assert attention_module.sparse_attn is owner.replacement_sparse
    assert getattr(attention, binding._OWNER) is owner
    monkeypatch.setattr(torch.npu, "synchronize", lambda value: None)
    owner.close()
    assert not owner.installed and owner.closed and not owner.devices and not owner.replacements
    assert owner._operator is owner._workspace_bytes is None
    assert attention_module.sparse_attn is setup.original and not hasattr(attention, binding._OWNER)


def test_grad_enabled_sink_is_read_only_in_an_explicit_inference_scope(setup):
    owner, attention, _ = setup.make()
    q, kv, sink, indices, scale = attention.operands
    attention.operands = q, kv, torch.nn.Parameter(sink), indices, scale
    before = attention.operands[2].detach().clone()
    owner.install()
    assert attention.forward(setup.x) is setup.original_result
    with torch.inference_mode():
        assert attention.forward(setup.x) is setup.candidate_result
    assert torch.equal(attention.operands[2], before)


def test_cpu_layout_operator_matches_the_original_read_only_attention(setup, monkeypatch):
    # The binding resolves this operator lazily, so the rest of the module is importable and testable
    # without it; only this equivalence test needs the real thing. Skipping rather than failing keeps
    # the suite green on a relic-core that has not landed the operator yet, which is the state this
    # binding was written against.
    pytest.importorskip("relic_core.kernels.ascend_sparse_attention_layout")
    from relic_core.kernels.ascend_sparse_attention_layout import sparse_attn_layout, layout_workspace_bytes
    from relic_core.kernels.ops import _sparse_attn_torch

    owner, attention, _ = setup.make()
    q, kv, sink, indices, scale = attention.operands
    q.fill_(0.5)
    indices[..., 1] = -7
    before = tuple(value.clone() for value in (q, kv, sink, indices))
    expected = _sparse_attn_torch(q, kv, sink, indices, scale)
    monkeypatch.setattr(owner, "_resolve_operator", lambda: (sparse_attn_layout, layout_workspace_bytes))
    owner.install()
    with torch.inference_mode():
        result = attention.forward(setup.x)
    assert torch.equal(result, expected)
    assert all(torch.equal(value, source) for value, source in zip((q, kv, sink, indices), before))
    assert owner.describe()["python_counters_include_capture_not_device_replay"] is True
