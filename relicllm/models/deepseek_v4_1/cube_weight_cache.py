"""Explicit bounded cache of exact FP4 I8x2 operands for native Cube experiments.

The original packed bank remains the source. This cache holds only recently used
projection operands, not a second full expert bank. Its owner must certify source
immutability and normal-BF16 representability before use. Production defaults do
not install it, and the native accumulation contract remains a separate experiment.
"""
from __future__ import annotations

from collections import OrderedDict

import torch

from . import cube_experiment as cube


class CubeWeightCache:
    def __init__(self,*,budget_bytes,certified_immutable_sources=False):
        if isinstance(budget_bytes,bool) or not isinstance(budget_bytes,int) or not 0<=budget_bytes<=4*2**30:
            raise ValueError("the experimental I8 operand cache is bounded to at most4GiB")
        if not certified_immutable_sources:
            raise ValueError("the owner must certify immutable original source operands")
        self.budget=budget_bytes
        self.entries=OrderedDict()
        self.bytes=0
        self.hits=self.misses=self.evictions=0

    @staticmethod
    def _token(tensor):
        try:version=tensor._version
        except RuntimeError:version=None
        storage=tensor.untyped_storage()
        return (storage._cdata,storage.data_ptr(),tensor.data_ptr(),tensor.storage_offset(),
                tuple(tensor.shape),tuple(tensor.stride()),str(tensor.dtype),str(tensor.device),version)

    def _weight(self,raw,scale,backend):
        key=self._token(raw),self._token(scale)
        if key in self.entries:
            entry=self.entries.pop(key)
            self.entries[key]=entry
            self.hits+=1
            return entry[2],entry[3]
        self.misses+=1
        _,_,decode,half=cube._core_ops()
        weight=decode(raw,backend=backend).t().contiguous()
        scales=half(scale,backend=backend).t().contiguous().to(torch.bfloat16)
        size=weight.numel()*weight.element_size()+scales.numel()*scales.element_size()
        if size<=self.budget:
            while self.entries and self.bytes+size>self.budget:
                _,old=self.entries.popitem(last=False)
                self.bytes-=old[4]
                self.evictions+=1
            # Strong source references prevent a freed/recycled pointer hit. Tensor
            # views of the same immutable arena intentionally have the same token.
            self.entries[key]=(raw,scale,weight,scales,size)
            self.bytes+=size
        return weight,scales

    def __call__(self,a,sa,b,sb,*,backend="auto"):
        raw_a=cube._checked_codes(a,"activation",encoding="fp8")
        raw_b=cube._checked_codes(b,"original packed FP4",encoding="fp4")
        k=raw_b.shape[1]*2
        raw_sa=cube._checked_scales(sa,"activation E8M0",k//32)
        raw_sb=cube._checked_scales(sb,"weight E8M0",k//32)
        if raw_a.shape!=(1,k) or raw_sb.shape[0]!=raw_b.shape[0]:
            raise ValueError("the cached Cube provider requires matching M1 original operands")
        if len({item.device for item in (raw_a,raw_sa,raw_b,raw_sb)})!=1:
            raise ValueError("all cached Cube operands must be on one device")
        chosen=cube._select_decode_backend(backend,raw_a.device)
        _,decode_a,_,_=cube._core_ops()
        activation=decode_a(raw_a,raw_sa,block_shape=(1,32),backend=chosen)
        weight,scales=self._weight(raw_b,raw_sb,chosen)
        if chosen=="torch":
            decoded=weight.t().float()*scales.t().float().repeat_interleave(32,dim=1)
            return (activation @ decoded.to(torch.bfloat16).t()).float()
        product=cube._npu_weight_quant_op()(activation,weight,scales,
                antiquant_offset=None,quant_scale=None,quant_offset=None,bias=None,
                antiquant_group_size=32,inner_precise=0)
        return product.float()

    def clear(self):
        count=len(self.entries)
        self.entries.clear()
        self.bytes=0
        return count

    def describe(self):
        return {"budget_bytes":self.budget,"retained_bytes":self.bytes,"entries":len(self.entries),
                "hits":self.hits,"misses":self.misses,"evictions":self.evictions,
                "source":"original packed FP4 with exact I8x2/half-scale operands",
                "native_accumulation_changed":True,"full_expert_bank_duplicate":False}
