"""Prepared AIV tables exactly match legacy code/scale/NZ indexing, with no weights."""
import hashlib
from types import SimpleNamespace

import pytest
import torch

from relic_core.kernels import ascendc_online_tile_cube as native


@pytest.mark.parametrize('group', [1,2,4,8])
@pytest.mark.parametrize('bits', [False,True])
def test_prepared_tables_reproduce_every_legacy_index_and_lookup_entry(group,bits):
    table = native._decode_table_cpu(group,bits).to(torch.int64) & 0xffffffff
    pairs,elements = 256*group,512*group
    assert table.dtype == torch.int64 and table.numel() == native._decode_table_words(group,bits)
    values = (0,0x3f00,0x3f80,0x3fc0,0x4000,0x4040,0x4080,0x40c0,
              0,0xbf00,0xbf80,0xbfc0,0xc000,0xc040,0xc080,0xc0c0)
    for index in range(256):
        code = (index+128)&255
        assert int(table[index]) == values[code&15] | (values[code>>4]<<16)
        scale = int(table[256+index])
        if bits:
            offset=((code-127)*128)&0xffff
            assert scale == ((offset|(offset<<16)) if 2<=code<=252 else 0)
        else:
            assert scale == ((code<<23) if 2<=code<=252 else 0x7fc00000)
    count=pairs if bits else elements
    assert torch.equal(table[512:512+count],torch.arange(count)//(16 if bits else 32)*4)
    nz=table[512+count:512+count+pairs]
    assert sorted(nz.tolist()) == list(range(0,pairs*4,4))
    for index in range(pairs):
        original_block=index//256
        half=(index%256)//128
        row=(index%128)//8
        pair=index%8
        assert int(nz[index]) == (original_block*256+row*16+half*8+pair)*4
    if bits:
        mask=table[512+count+pairs:768+count+pairs]
        poison=table[768+count+pairs:1024+count+pairs]
        for index in range(256):
            code=(index+128)&255
            assert int(mask[index]) == ((0xffff if code&7 else 0)|(0xffff0000 if code&0x70 else 0))
            assert int(poison[index]) == (0 if 2<=code<=252 else 0x7fc07fc0)


class Query:
    def __init__(self,value):self.value=value
    def __call__(self):return self.value


class Launch:
    def __call__(self,*args):pytest.fail('metadata tests must not launch a native kernel')


def operator(tmp_path,monkeypatch,*,enabled=1,words=3584,launch=True,rows=16):
    path=tmp_path/'metadata-only.so'
    path.write_bytes(b'verified fake native table ABI; never executed')
    symbols=dict(relic_online_tile_cube_launch=Launch(),relic_online_tile_cube_k_blocks_per_tile=Query(4),
                 relic_online_tile_cube_prepared_tables=Query(enabled),relic_online_tile_cube_table_words=Query(words),
                 relic_online_tile_cube_output_rows=Query(rows))
    if launch:symbols['relic_online_tile_cube_launch_tables']=Launch()
    monkeypatch.setattr(native.ctypes,'CDLL',lambda _:SimpleNamespace(**symbols))
    return native.AscendCTileCubeConsumer(path,sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_prepared_table_launch_abi_is_checked_and_metadata_is_small(tmp_path,monkeypatch):
    op=operator(tmp_path,monkeypatch)
    assert len(op._launch.argtypes)==15 and len(op._launch_tables.argtypes)==16
    assert op.describe()['decode_table_bytes_per_device']==14336
    assert op.describe()['whole_matrix_expanded_weight_scratch_bytes']==0
    assert op.describe()['decode_table_devices_prepared']==[]


@pytest.mark.parametrize('kwargs',[dict(enabled=2),dict(words=3583),dict(launch=False)])
def test_inconsistent_native_table_policy_or_pointer_abi_is_refused(tmp_path,monkeypatch,kwargs):
    with pytest.raises(ValueError):operator(tmp_path,monkeypatch,**kwargs)


def test_legacy_native_table_policy_does_not_prepare_cpu_metadata(tmp_path,monkeypatch):
    op=operator(tmp_path,monkeypatch,enabled=0,launch=False)
    op.prepare_device_metadata('cpu')
    assert not op._decode_tables and op._cpu_decode_table is None
    assert op.describe()['decode_table_bytes_per_device']==0


def test_enabled_native_table_metadata_cannot_use_a_cpu_fallback(tmp_path,monkeypatch):
    op=operator(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='explicit NPU'):
        op.prepare_device_metadata('cpu')


@pytest.mark.parametrize('group,bits',[(True,False),(3,False),(4,1),(4,None)])
def test_prepared_geometry_and_direct_bit_flag_are_not_ambiguous(group,bits):
    with pytest.raises(ValueError):native._decode_table_cpu(group,bits)


@pytest.mark.parametrize('group', [1,2,4])
def test_n32_nz_order_reads_original_n16_banks_without_repacking(group):
    from relic_core.kernels.online_tile_cube import pack_expert_tile_bank
    from relic_core.kernels.unpack import unpack_fp4_to_int8
    k=32*group
    generator=torch.Generator().manual_seed(320+group)
    code=torch.randint(0,256,(1,32,k//2),generator=generator,dtype=torch.uint8)
    scales=torch.full((1,32,group),127,dtype=torch.uint8)
    packed,_=pack_expert_tile_bank(code,scales)
    raw=packed[0].flatten().long()
    tables=native._decode_table_cpu(group,False,32).long()&0xffffffff
    pairs,elements=32*16*group,32*32*group
    source=tables[(raw+128)&255]
    nz_indices=tables[512+elements:512+elements+pairs]//4
    actual=source[nz_indices]
    dense=(unpack_fp4_to_int8(code,backend='torch').float()*.5).bfloat16()[0]
    expected=dense.view(torch.int16).reshape(2,16,k//16,16).permute(2,0,1,3).contiguous().flatten().long()&0xffff
    expected_pairs=expected[0::2]|(expected[1::2]<<16)
    assert torch.equal(actual,expected_pairs)


def test_n32_ring_geometry_and_original_bank_rows_are_explicit(tmp_path,monkeypatch):
    op=operator(tmp_path,monkeypatch,rows=32,words=6656)
    assert op.output_rows==32 and op.ring_elements==4096
    assert op.describe()['decode_table_bytes_per_device']==26624
    assert op.describe()['source_checkpoint_tile_rows_unchanged']==16


@pytest.mark.parametrize('rows', [0,17,64,True])
def test_unbounded_or_nonintegral_native_output_tiles_are_refused(tmp_path,monkeypatch,rows):
    with pytest.raises(ValueError,match='output-row geometry'):
        operator(tmp_path,monkeypatch,rows=rows)


def test_wide_tile_workspace_changes_only_bounded_ring_capacity():
    from relic_core.kernels.ascendc_tile_moe import make_tile_moe_workspace
    old=make_tile_moe_workspace(5120,1152,6,device='cpu',group_k=4)
    new=make_tile_moe_workspace(5120,1152,6,device='cpu',group_k=4,weight_rows=32)
    assert new.weight_storage.shape==(24,2,4096)
    assert new.nbytes-old.nbytes==196608
    assert new.gate_accumulation.shape==old.gate_accumulation.shape==(6,1152)
    assert new.down_accumulation.shape==old.down_accumulation.shape==(6,5120)
