#!/usr/bin/env bash
# Reuse the proven ABI-matched interpreter read-only; all task outputs are new.
source <workspace>/relic-online-unpack-20261117-0534/source/kernel-env-scripts/env311.sh
export RELIC_EXPERT_ROOT=<workspace>/relic-expert-gemm-20261117-1046
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$RELIC_EXPERT_ROOT/source/relic-core:$RELIC_EXPERT_ROOT/source/RelicLLM:/usr/local/Ascend/cann-9.2.0/python/site-packages:/usr/local/Ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe"
export TMPDIR="$RELIC_EXPERT_ROOT/tmp"
export HOME="$RELIC_EXPERT_ROOT/home"
export TRITON_CACHE_DIR="$RELIC_EXPERT_ROOT/cache/triton"
export ASCEND_CACHE_PATH="$RELIC_EXPERT_ROOT/cache/ascend"
export ASCEND_WORK_PATH="$RELIC_EXPERT_ROOT/cache/ascend-work"
export ASCEND_PROCESS_LOG_PATH="$RELIC_EXPERT_ROOT/logs/ascend"
export ASCEND_RT_VISIBLE_DEVICES=0
export RELIC_EXPERT_TEST_DEVICE=npu:0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
