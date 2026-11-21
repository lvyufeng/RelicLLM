"""Prime existing exact activation kernels serially before an eight-rank trial."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import torch
import torch_npu
from relic_core.kernels.fast_fused_fp8_activation_decode import quantize_decode_fp8_fast


@torch.inference_mode()
def main():
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(90128)
    records = []
    for shape in ((1, 6144), (6, 1152)):
        for dtype in (torch.bfloat16, torch.float32):
            host = torch.randn(shape, generator=generator).to(dtype)
            expected = quantize_decode_fp8_fast(host, backend='torch')
            x = host.to('npu:0')
            print(json.dumps({'priming_existing_kernel': list(shape), 'input_dtype': str(dtype),
                              'return_codes': False, 'not_a_performance_measurement': True}), flush=True)
            start = time.perf_counter()
            value = quantize_decode_fp8_fast(x, backend='triton')
            torch.npu.synchronize()
            if not torch.equal(value.cpu().view(torch.int16), expected.view(torch.int16)):
                raise RuntimeError('the existing fused activation differs from its original Torch oracle')
            records.append({'shape': list(shape), 'dtype': str(dtype), 'decoded_BF16_bit_differences': 0,
                            'initial_compile_and_execute_seconds': time.perf_counter() - start})
    source = Path(inspect.getsourcefile(inspect.unwrap(quantize_decode_fp8_fast)))
    report = {'cases': records, 'module_source': str(source),
              'module_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'Triton_cache': os.environ.get('TRITON_CACHE_DIR'), 'not_model_TPS': True}
    with (Path(os.environ['RELIC_PERF_OUTPUT']) / 'activation-cache-priming.json').open('x') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
