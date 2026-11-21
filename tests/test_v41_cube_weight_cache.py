"""Source identity, bounded retention and exact operand-cache composition."""
import pytest
import torch

from relicllm.models.deepseek_v4_1 import cube_experiment as cube
from relicllm.models.deepseek_v4_1.cube_weight_cache import CubeWeightCache


def values():
    a=torch.full((1,64),0x38,dtype=torch.uint8)
    sa=torch.full((1,2),127,dtype=torch.uint8)
    b=torch.full((32,32),0x11,dtype=torch.uint8)
    sb=torch.full((32,2),127,dtype=torch.uint8)
    return a,sa,b,sb


def test_cache_preserves_the_natural_candidate_result_and_reuses_arena_views():
    a,sa,b,sb=values()
    expected=cube.cube_fp4_packed_gemv(a,sa,b,sb,backend="torch")
    cache=CubeWeightCache(budget_bytes=8192,certified_immutable_sources=True)
    assert torch.equal(cache(a,sa,b,sb,backend="torch"),expected)
    assert torch.equal(cache(a,sa,b.view_as(b),sb.view_as(sb),backend="torch"),expected)
    assert cache.describe()["hits"]==1 and cache.describe()["misses"]==1
    assert cache.describe()["retained_bytes"]<=8192


def test_mutated_sources_do_not_hit_an_old_version():
    a,sa,b,sb=values()
    cache=CubeWeightCache(budget_bytes=8192,certified_immutable_sources=True)
    before=cache(a,sa,b,sb,backend="torch")
    b.fill_(0x22)
    after=cache(a,sa,b,sb,backend="torch")
    assert not torch.equal(before,after)
    assert cache.misses==2 and cache.hits==0


def test_cache_budget_bounds_entries_and_zero_budget_caches_nothing():
    a,sa,b,sb=values()
    for budget in (0,2200):
        cache=CubeWeightCache(budget_bytes=budget,certified_immutable_sources=True)
        for _ in range(3):cache(a,sa,b.clone(),sb.clone(),backend="torch")
        assert cache.bytes<=budget
        assert len(cache.entries)==(0 if budget==0 else 1)
        cache.clear()
        assert cache.bytes==0 and len(cache.entries)==0


def test_cache_is_never_enabled_without_a_certified_immutable_source_owner():
    with pytest.raises(ValueError,match="certify"):CubeWeightCache(budget_bytes=1024)
    with pytest.raises(ValueError,match="4GiB"):
        CubeWeightCache(budget_bytes=5*2**30,certified_immutable_sources=True)
