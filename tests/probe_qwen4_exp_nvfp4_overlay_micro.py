"""Qualify additive operators against the actually loaded legacy CUDA binary before timing."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overlay-so',type=Path,required=True)
    parser.add_argument('--core-tests',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import torch
    import pytest
    from relic_core.kernels.cuda_loader import load_cuda_kernel
    from relicllm.models.qwen4_exp.nvfp4_overlay import install_nvfp4_overlay
    if not torch.cuda.is_available():
        raise RuntimeError('the independent operator oracle requires a leased CUDA device')
    previous=os.environ.get('POCKETLLM_QWEN4_NVFP4_OVERLAY_SHA256')
    old_cwd=Path.cwd();old_argv=list(sys.argv);old_path=list(sys.path)
    try:
        with install_nvfp4_overlay(load_cuda_kernel(),args.overlay_so) as handle:
            try:
                os.environ['POCKETLLM_QWEN4_NVFP4_OVERLAY_SHA256']=handle.identity['overlay_sha256']
                print('operator_overlay_identity',handle.identity,flush=True)
                sys.path.insert(0,str(args.core_tests));os.chdir(args.core_tests.parent)
                files=[str(args.core_tests/name) for name in ['test_nvfp4_exact_store.py','test_nvfp4_exact_compact.py','test_nvfp4_exact_fast.py']]
                rc=pytest.main(files+['-q'])
                if rc:return int(rc)
                import bench_nvfp4_exact_launches
                sys.argv=['bench_nvfp4_exact_launches.py','--variant','store','--rows','1,4,512',
                          '--banks','8','--samples','7','--output',str(args.output)]
                return int(bench_nvfp4_exact_launches.main() or 0)
            finally:
                torch.cuda.synchronize()
    finally:
        os.chdir(old_cwd);sys.argv=old_argv;sys.path[:]=old_path
        if previous is None:
            os.environ.pop('POCKETLLM_QWEN4_NVFP4_OVERLAY_SHA256',None)
        else:
            os.environ['POCKETLLM_QWEN4_NVFP4_OVERLAY_SHA256']=previous


if __name__=='__main__':raise SystemExit(main())
