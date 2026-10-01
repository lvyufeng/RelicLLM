"""One door to the operator surface, and the two things the door has to keep true.

The seam exists so that "which hardware am I on" is asked once instead of being answered seventy
times by the spelling of a call. Two claims make that a seam rather than a rewrite, and both are
tested here:

* **On CUDA it is a no-op.** `load_ops()` hands back *the same object* `load_cuda_kernel()` does --
  identity, not equivalence -- so the `hasattr(ops, "<binding>")` probes and the `None` fallbacks the
  sixteen call sites already use behave exactly as they did before this module existed.
* **An unimplemented platform answers `None`.** Not an error: the probing call sites were written
  for exactly that answer, and their docstring already names Ascend as the reason. A platform with no
  provider is the case they handle, so the seam arriving before the provider is a supported state
  rather than a half-finished one.

Nothing here needs hardware of the kind it is proving, which is the point -- the same discipline
`test_device_platform.py` uses, and for the same reason.
"""

from __future__ import annotations

import pytest

from src.runtime import ops as ops_plane


class _FakeProvider:
    """A provider that answers for some of :data:`~src.runtime.ops.BINDINGS` and not the rest.

    Deliberately partial. A provider is usually partial while it is being built, and the useful
    question about one is which parts it has, not whether it is finished.
    """

    def __init__(self, *bindings: str) -> None:
        for name in bindings:
            setattr(self, name, lambda *args, **kwargs: None)


@pytest.fixture
def clean_registry():
    """Take the registry out and put it back, so a test cannot leak a provider into the next one."""
    before = dict(ops_plane._PROVIDERS)
    yield
    ops_plane._PROVIDERS.clear()
    ops_plane._PROVIDERS.update(before)


def _extension():
    """The built CUDA extension, or a skip. Only the tests that inspect it need it."""
    from relic_core.kernels.cuda_loader import load_cuda_kernel

    module = load_cuda_kernel()
    if module is None:
        pytest.skip("the `cuda_kernel` extension is not built for this interpreter")
    return module


# ------------------------------------------------------------------- the CUDA path is unchanged


def test_cuda_still_hands_back_relic_cores_extension_itself() -> None:
    """Identity, not equivalence: the object is the one `load_cuda_kernel()` returns.

    This is the whole of the no-op claim. A wrapper that forwarded every attribute would pass a
    behavioural test and still break `hasattr` on a binding the extension does not export, which is
    the check half these call sites make.
    """
    from relic_core.kernels.cuda_loader import load_cuda_kernel

    _extension()  # skip if it is not built
    assert ops_plane.load_ops("cuda") is load_cuda_kernel()


def test_the_default_platform_is_the_one_this_host_has() -> None:
    """`load_ops()` asks the same question `--device auto` does, and gets the same answer.

    That agreement is the point of the default: a launch picks a platform and the operator lookup
    finds the same one, rather than a run resolving to Ascend and then asking CUDA for its kernels.
    """
    from src.runtime.device import probe_accelerator

    platform = probe_accelerator().platform
    assert ops_plane.load_ops() is ops_plane.load_ops(platform)


# ------------------------------------------------------------------- a second provider


def test_a_registered_provider_is_what_that_platform_gets(clean_registry) -> None:
    """The seam's whole purpose: a platform arrives through registration, not through a call site.

    Nothing in the runtime passes a platform to `load_ops()` -- the call sites are unchanged from
    before this module existed. What changed is that the answer can now depend on the hardware.
    """
    provider = _FakeProvider("mimo_rope_rows")
    ops_plane.register_provider("ascend", lambda: provider)

    assert ops_plane.load_ops("ascend") is provider
    assert ops_plane.registered_platforms() == ("ascend",)
    # And CUDA is untouched by a neighbour's registration.
    assert ops_plane.load_ops("cuda") is not provider


def test_an_unimplemented_platform_answers_none_rather_than_raising(clean_registry) -> None:
    """`None`, because that is the answer the call sites were already written to handle.

    `models/xing4_0/hyper_connection.py` states the posture: *"`None` rather than raising: this is
    one kernel of many in a tree that also builds for Ascend, and a forward pass that falls back to
    the eager arithmetic is correct, just slower."* Raising here would turn every probe-and-fall-back
    site into a crash on the platform the seam exists for.
    """
    assert ops_plane.load_ops("ascend") is None
    assert ops_plane.load_ops("cpu") is None
    assert ops_plane.registered_platforms() == ()


def test_a_registration_can_be_taken_back(clean_registry) -> None:
    """What makes the seam testable, and what stops a provider leaking across tests."""
    ops_plane.register_provider("ascend", lambda: _FakeProvider())
    assert ops_plane.load_ops("ascend") is not None
    ops_plane.unregister_provider("ascend")
    assert ops_plane.load_ops("ascend") is None
    # Idempotent, so a fixture can clean up without knowing whether it registered anything.
    ops_plane.unregister_provider("ascend")


def test_a_later_registration_replaces_an_earlier_one(clean_registry) -> None:
    first, second = _FakeProvider(), _FakeProvider()
    ops_plane.register_provider("ascend", lambda: first)
    ops_plane.register_provider("ascend", lambda: second)
    assert ops_plane.load_ops("ascend") is second


def test_the_default_platform_is_asked_once_rather_than_on_every_call(monkeypatch) -> None:
    """`load_ops()` with no argument is a decode-loop call, so the host probe behind it is latched.

    The cost is real and it lands on this box rather than on an Ascend one. `probe_accelerator()`
    measures about fifty microseconds here, and almost all of it is the *failing* `import torch_npu`
    inside `_npu_available` -- a failure CPython does not cache, so it is paid again on every call.
    Before this was latched, `load_ops()` measured 57-70 us against the 0.045 us of the cached
    global it replaced, and `models/deepseek_v4/runtime.py`'s `rotate_activation` and
    `models/minimax_m2/architecture.py`'s `_apply_rope` resolve the module per layer per step.

    The assertion is on the *count*, not on the timing: a test that measured microseconds would be
    a flake on a loaded host, and what is being pinned is the number of times the process asks a
    question whose answer cannot change.

    **Latching the platform is the whole of the cache.** The provider lookup is not cached, which is
    why `test_a_later_registration_replaces_an_earlier_one` needs no invalidation: the two tests
    together say the seam is cheap *and* live.
    """
    real = ops_plane.probe_accelerator
    asked: list[None] = []

    def counting():
        asked.append(None)
        return real()

    monkeypatch.setattr(ops_plane, "probe_accelerator", counting)
    ops_plane._host_platform.cache_clear()
    try:
        for _ in range(5):
            ops_plane.load_ops()
    finally:
        # Leave the cache holding the real answer, so the next test does not inherit the stub.
        ops_plane._host_platform.cache_clear()
    assert len(asked) == 1, (
        f"the host was asked {len(asked)} times for five calls. The answer is a property of the "
        f"process, and the call sites that use the default run it per layer per step"
    )


# ------------------------------------------------------------------- the declared surface


def test_the_binding_list_is_a_set_and_names_only_real_bindings() -> None:
    """No duplicates, and -- where the extension is built -- nothing the extension cannot answer.

    The second half is the one that matters over time. `BINDINGS` is a checked-in list describing
    what a provider for a new platform has to supply; a binding that was renamed in relic-core and
    not here would leave the list quietly claiming a name nothing exports, and a provider author
    would build against a contract that no longer describes the runtime.
    """
    assert len(set(ops_plane.BINDINGS)) == len(ops_plane.BINDINGS)
    extension = _extension()
    unknown = sorted(name for name in ops_plane.BINDINGS if not hasattr(extension, name))
    assert unknown == [], f"BINDINGS names bindings the extension does not export: {unknown}"


def test_missing_bindings_reports_the_difference_in_declaration_order() -> None:
    """A diagnostic, asked the useful way round: which parts are absent, not whether it is done."""
    partial = _FakeProvider("mimo_rope_rows", "q8_0_gemm_forward")
    missing = ops_plane.missing_bindings(partial)

    assert "mimo_rope_rows" not in missing
    assert "q8_0_gemm_forward" not in missing
    assert set(missing) == set(ops_plane.BINDINGS) - {"mimo_rope_rows", "q8_0_gemm_forward"}
    # Declaration order, so a report reads like the list it came from.
    assert missing == tuple(n for n in ops_plane.BINDINGS if n in set(missing))
    # An operator module with everything answers with nothing missing.
    assert ops_plane.missing_bindings(_FakeProvider(*ops_plane.BINDINGS)) == ()
