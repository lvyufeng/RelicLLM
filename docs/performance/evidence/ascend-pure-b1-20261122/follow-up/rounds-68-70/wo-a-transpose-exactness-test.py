"""Is the cached-transpose einsum bit-identical to the original, and does the patch bind correctly?"""
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, '/data1/relic-v41-pure-b1-20261122-1325/source')
sys.path.insert(0, '/data1/relic-v41-pure-b1-20261122-1325/source/RelicLLM')
sys.path.insert(0, '/data1/relic-v41-pure-b1-20261122-1325/source/relic-core')

torch.npu.set_device(0)
device = torch.device('npu', 0)

groups, o_lora_rank, d = 2, 1024, 2048
bsz, seqlen = 1, 1

ok = True
for trial in range(6):
    generator = torch.Generator().manual_seed(100 + trial)
    o = torch.randn((bsz, seqlen, groups, d), generator=generator, dtype=torch.bfloat16).to(device)
    weight = torch.randn((groups * o_lora_rank, d), generator=generator, dtype=torch.bfloat16).to(device)

    wo_a = weight.view(groups, o_lora_rank, -1)
    original = torch.einsum("bsgd,grd->bsgr", o, wo_a)

    wo_a_t = weight.view(groups, o_lora_rank, -1).transpose(1, 2).contiguous()
    cached = torch.einsum("bsgd,gdr->bsgr", o, wo_a_t)

    same = torch.equal(original.cpu(), cached.cpu())
    ok = ok and same
    if not same:
        difference = (original.cpu() != cached.cpu())
        print('trial %d MISMATCH: %d of %d differ, max abs %s'
              % (trial, int(difference.sum()), difference.numel(),
                 (original.cpu().float() - cached.cpu().float()).abs().max()))
    else:
        print('trial %d exact' % trial)

print()
print('einsum forms bit-identical:', ok)

print()
print('=== does the patch install on exactly one class? ===')
import ascend_wo_a_cached_transpose as patcher
from relicllm.models.deepseek_v4_1 import attention

owner = patcher.install()
print('owner:', owner)
patched = [c.__name__ for c in (attention,) if True]
import importlib
module = importlib.import_module('relicllm.models.deepseek_v4_1.attention')
replaced = []
for name in dir(module):
    candidate = getattr(module, name, None)
    if isinstance(candidate, type) and candidate.__dict__.get('_relic_original_forward') is not None:
        replaced.append(name)
        assert candidate.forward is not candidate.__dict__['_relic_original_forward']
print('classes rebound:', replaced)
patcher.uninstall()
restored = [n for n in dir(module)
            if isinstance(getattr(module, n, None), type)
            and getattr(module, n).__dict__.get('_relic_original_forward') is not None]
print('restored cleanly:', not restored)
raise SystemExit(0 if ok and len(replaced) == 1 and not restored else 1)
