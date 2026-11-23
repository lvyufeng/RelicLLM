"""Bind the fused sparse-indexer selection into the decode path, reversibly and only on request.

`Indexer.forward` builds the sliding-window block selection as an inline expression: mask the entries
the query cannot reach, sort them, map the sentinel back to `-1`. On the Ascend build each of those
steps is an **AI CPU** operator, and an AI CPU operator holds no cube or vector unit while serialising
against the stream -- 31 operators a forward, 2762 us, 3.5% of the step, almost all of it fixed
dispatch cost rather than work.

The selection itself is not a function, it is one expression inside a long method, so there is nothing
to wrap. This binding reads the method's own source, swaps that one expression for the fused call, and
binds the recompiled method back to the class. **The model file on disk is not touched**, so its hash
and everything derived from it are unchanged, and reverting is one assignment.

Two rules make that safe enough to ship:

* If the anchor is not found **exactly once**, install raises. A silent fallback would let a benchmark
  report a gain that came from code which never ran; refusing is the only honest option.
* The 2048-wide second site is **deliberately left alone**. The fused selection measured 0.66x there --
  slower than the four operators it would remove -- so it stays on the reference path rather than being
  traded for a regression.

Nothing is reached by importing this module. `install()` is explicit, refuses a second owner that is not
its own, and `close()` restores the exact method object that was replaced.
"""
from __future__ import annotations

import inspect
import textwrap
import threading

#: The reference expression at the replaced site, transcribed with its own indentation removed.
ANCHOR = (
    "    idxs = torch.where(values > -torch.inf, keys + offset, tail).sort(dim=-1).values\n"
    "    idx = torch.where(idxs < tail, idxs, -1).int()"
)

#: The fused call that replaces it. Bound into the method's namespace at install time.
REPLACEMENT = "    idx = fused_masked_sort_trailing(values, keys + offset, width, tail)"

_INSTALLED_FLAG = "_relic_indexer_sort_installed"
_OWNER = "_relic_indexer_sort_owner"
_ORIGINAL = "_relic_indexer_sort_original"


class IndexerSortBinding:
    """The fused selection, installed into `Indexer.forward`, with what it replaced recorded."""

    def __init__(self, module, original, patched):
        self._module = module
        self._original = original
        self._patched = patched
        self._thread = threading.get_ident()
        self.install_count = 1

    def describe(self):
        return {
            "installed": True,
            "awareness": "relic_core.kernels.ascend_indexer_masked_sort",
            "site_replaced": "the stream-prefix selection inside Indexer.forward",
            "site_left_alone": "the 2048-wide block selection (fused form measured 0.66x there)",
            "anchor_replacements": 1,
            "model_source_touched": False,
            "numerical_or_performance_acceptance_claimed": False,
        }

    def close(self):
        """Restore the original method object, refusing from a thread that does not own the binding."""
        if threading.get_ident() != self._thread:
            raise RuntimeError("the indexer sort binding has to be closed from the thread that installed it")
        # Check the live method, not just the saved one: if something replaced `forward`
        # behind this binding, restoring would silently discard that replacement.
        if self._module.Indexer.forward is not self._patched:
            raise RuntimeError("the replaced method is no longer the one this binding installed")
        if getattr(self._module.Indexer, _ORIGINAL, None) is not self._original:
            raise RuntimeError("the saved original method is not the one this binding recorded")
        self._module.Indexer.forward = self._original
        for name in (_ORIGINAL, _OWNER, _INSTALLED_FLAG):
            if hasattr(self._module.Indexer, name):
                delattr(self._module.Indexer, name)
        return True


def _dedented_source(function):
    """Dedented source, because `getsource` returns a method with its class indentation."""
    return textwrap.dedent(inspect.getsource(function))


def install():
    """Replace the stream-prefix selection in `Indexer.forward` and prove the swap took."""
    from relicllm.models.deepseek_v4_1 import attention

    existing = getattr(attention.Indexer, _OWNER, None)
    if existing is not None:
        # A second owner would stack a second patch on the first with no way to unwind it.
        raise RuntimeError("the indexer sort binding is already installed")
    if not hasattr(attention.Indexer, "forward"):
        raise RuntimeError("Indexer has no forward to bind the fused selection into")

    from relic_core.kernels.ascend_indexer_masked_sort import masked_sort_trailing_with_tail

    original = attention.Indexer.forward
    source = _dedented_source(original)
    found = source.count(ANCHOR)
    if found != 1:
        raise RuntimeError("indexer selection anchor found %d times, expected exactly 1" % found)

    namespace = dict(attention.__dict__)
    namespace["fused_masked_sort_trailing"] = masked_sort_trailing_with_tail
    exec(compile(source.replace(ANCHOR, REPLACEMENT, 1), "<indexer-sort-fused>", "exec"), namespace)
    patched = namespace.get("forward")
    if patched is None:
        raise RuntimeError("the swapped source did not define a method named forward")
    if patched is original:
        raise RuntimeError("the patched method is the original object; the swap did not take")

    attention.Indexer.forward = patched
    setattr(attention.Indexer, _ORIGINAL, original)
    owner = IndexerSortBinding(attention, original, patched)
    setattr(attention.Indexer, _OWNER, owner)
    setattr(attention.Indexer, _INSTALLED_FLAG, True)
    return owner