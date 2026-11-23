"""Keep `wo_a` in the layout the einsum wants, so the constant weight is not transposed every step.

`attention.py:1688` reads the o-projection weight as `[groups, o_lora_rank, d]` and then einsums over
`d`, so the framework materialises a transpose of that weight into `[groups, d, o_lora_rank]` before the
matmul — every layer, every token. The weight never changes during a decode, so the transpose is repeated
work: on the registered lane it shows up as `aclnnBatchMatMul_TransposeAiCore_Transpose`, 48 calls and
1,171.6 us a forward, dominated by the `1024,4096` to `4096,1024` shape.

Transposing once per module and contracting the same axis in the same order computes the same values: the
sum is still over `d`, ascending, over the same products. Only the storage layout of a constant changes.
Nothing about the checkpoint, the activation grid, the FP32 tensor-parallel reduction or the collective
order is touched.

The patch is applied to the method's own source at run time, so the model file on disk is unchanged and
reverting is one assignment. If the anchor is not found in exactly one method this raises rather than
silently doing nothing.
"""
from __future__ import annotations

import inspect
import textwrap

ANCHOR = """    wo_a = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, -1)
    o = torch.einsum("bsgd,grd->bsgr", o, wo_a)"""
REPLACEMENT = """    wo_a_t = getattr(self, "_relic_wo_a_cached", None)
    if wo_a_t is None:
        wo_a_t = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, -1).transpose(1, 2).contiguous()
        self._relic_wo_a_cached = wo_a_t
    o = torch.einsum("bsgd,gdr->bsgr", o, wo_a_t)"""
OWNER = '_relic_wo_a_transpose_owner'


def _candidates(module):
    """Every class in the model module that defines a `forward`, so the anchor locates itself."""
    for name in dir(module):
        candidate = getattr(module, name, None)
        if not isinstance(candidate, type):
            continue
        function = candidate.__dict__.get('forward')
        if function is None:
            continue
        try:
            source = textwrap.dedent(inspect.getsource(function))
        except (OSError, TypeError):
            continue
        yield candidate, function, source


def install(module_name='relicllm.models.deepseek_v4_1.attention'):
    import importlib

    module = importlib.import_module(module_name)
    existing = getattr(module, OWNER, None)
    if existing is not None:
        return existing
    matches = [(cls, fn, src) for cls, fn, src in _candidates(module) if ANCHOR in src]
    if len(matches) != 1:
        raise RuntimeError('the wo_a einsum anchor appears in %d methods, expected exactly 1: %s'
                           % (len(matches), [cls.__name__ for cls, _, _ in matches]))
    cls, original, source = matches[0]
    patched_source = textwrap.dedent(source).replace(ANCHOR, REPLACEMENT, 1)
    namespace = dict(module.__dict__)
    namespace['torch'] = __import__('torch')
    exec(compile(patched_source, '<wo-a-cached-transpose>', 'exec'), namespace)
    patched = namespace['forward']
    if patched is original:
        raise RuntimeError('the patched method is the original object; the swap did not take')
    cls._relic_original_forward = original
    cls.forward = patched
    owner = {'installed': True, 'class': cls.__name__,
             'site': 'attention.py:1688-1689',
             'change': 'constant wo_a weight pre-transposed once per module instead of every step',
             'numerically_identical': 'same products, same ascending reduction over d, constant storage '
                                      'layout only',
             'extra_resident_bytes_per_layer': 4 * 1024 * 4}
    setattr(module, OWNER, owner)
    return owner


def uninstall(module_name='relicllm.models.deepseek_v4_1.attention'):
    import importlib

    module = importlib.import_module(module_name)
    for name in dir(module):
        candidate = getattr(module, name, None)
        if isinstance(candidate, type):
            original = candidate.__dict__.get('_relic_original_forward')
            if original is not None:
                candidate.forward = original
                delattr(candidate, '_relic_original_forward')
    if hasattr(module, OWNER):
        delattr(module, OWNER)
