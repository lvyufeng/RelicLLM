"""Bind the fused selection into `Indexer._stream_prefix` without editing the model source.

The site to replace is an inline expression inside a long method, not a function, so there is nothing to
wrap. Instead the method's own source is read, the one expression is swapped, and the result is compiled
and bound to the class. The model file on disk is not touched, so its hash is unchanged and reverting is
a single assignment.

Only site A (`attention.py:1173`, inside `Indexer.forward`) is replaced. Site B (`:1326`, a 2048-wide block selection) measured
0.66x with the rank kernel - slower than the four operators it would remove - so it is deliberately left
alone rather than traded for a regression.

If the anchor is not found exactly once this raises. A silent fallback would let a benchmark report a
gain that came from code which never ran.
"""
from __future__ import annotations

from pathlib import Path

ANCHOR = """    idxs = torch.where(values > -torch.inf, keys + offset, tail).sort(dim=-1).values
    idx = torch.where(idxs < tail, idxs, -1).int()"""
REPLACEMENT = """    idx = fused_masked_sort_trailing(values, keys + offset, width, tail)"""
OWNER = '_relic_indexer_sort_owner'


def install():
    """Replace site A in the indexer's stream prefix selection, and prove the swap took."""
    from relicllm.models.deepseek_v4_1 import attention

    existing = getattr(attention, OWNER, None)
    if existing is not None:
        return existing
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import ascend_indexer_sort_fused as fused

    original = attention.Indexer.forward
    source = _get_source(original)
    if source.count(ANCHOR) != 1:
        raise RuntimeError('indexer selection anchor found %d times, expected exactly 1'
                           % source.count(ANCHOR))
    patched_source = source.replace(ANCHOR, REPLACEMENT, 1)
    namespace = dict(attention.__dict__)
    namespace['fused_masked_sort_trailing'] = fused.masked_sort_trailing_with_tail
    exec(compile(patched_source, '<indexer-sort-fused>', 'exec'), namespace)
    patched = namespace['forward']
    if patched is original:
        raise RuntimeError('the patched method is the original object; the swap did not take')
    attention.Indexer.forward = patched
    attention.Indexer._relic_original_indexer_forward = original
    owner = {'installed': True, 'anchor_replacements': 1, 'site_replaced': 'attention.py:1173',
             'site_left_alone': 'attention.py:1326 (2048-wide, rank kernel measured 0.66x)'}
    setattr(attention, OWNER, owner)
    return owner


def uninstall():
    from relicllm.models.deepseek_v4_1 import attention

    original = getattr(attention.Indexer, '_relic_original_indexer_forward', None)
    if original is not None:
        attention.Indexer.forward = original
    if hasattr(attention, OWNER):
        delattr(attention, OWNER)


def _get_source(function):
    """Dedented source, because `getsource` returns a class method with its class indentation and
    `exec` at module level rejects it."""
    import inspect
    import textwrap

    return textwrap.dedent(inspect.getsource(function))
