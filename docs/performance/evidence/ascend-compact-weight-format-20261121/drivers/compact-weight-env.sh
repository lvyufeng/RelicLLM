#!/usr/bin/env bash
# Reuse established ABI-matched dependencies; keep all new writes private.
source <private-control-root>/env.sh
export RELIC_COMPACT_ROOT=<private-compact-root>
export PYTHONPATH=<private-dependencies-root>/python-packages:$RELIC_COMPACT_ROOT/source/relic-core:$RELIC_COMPACT_ROOT/source/RelicLLM:/usr/local/Ascend/cann-9.2.0/python/site-packages:/usr/local/Ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe
export TMPDIR=$RELIC_COMPACT_ROOT/tmp
export HOME=$RELIC_COMPACT_ROOT/home
export TRITON_CACHE_DIR=$RELIC_COMPACT_ROOT/cache/triton
export ASCEND_CACHE_PATH=$RELIC_COMPACT_ROOT/cache/ascend
export ASCEND_WORK_PATH=$RELIC_COMPACT_ROOT/cache/ascend-work
export ASCEND_PROCESS_LOG_PATH=$RELIC_COMPACT_ROOT/logs/ascend
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
