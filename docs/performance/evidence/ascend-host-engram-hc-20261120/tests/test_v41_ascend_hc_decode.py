"""CPU logic contracts for reversible per-block HC decode-fold installation."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from relicllm.models.deepseek_v4_1.ascend_hc_decode import FusedHCDecodeFold


class Block(nn.Module):
    hc_mult = 4

    def hc_pre(self, x, pre_mix):
        return (pre_mix.unsqueeze(-1) * x.float()).sum(dim=2).bfloat16()

    def hc_post(self, x, residual, post, comb):
        return (post.unsqueeze(-1) * x.unsqueeze(-2)
                + (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(dim=2)).bfloat16()


def tensors(tokens=1):
    generator = torch.Generator().manual_seed(13)
    x = torch.randn(1, tokens, 64, generator=generator).bfloat16()
    residual = torch.randn(1, tokens, 4, 64, generator=generator).bfloat16()
    pre = torch.randn(1, tokens, 4, generator=generator)
    post = torch.randn(1, tokens, 4, generator=generator)
    comb = torch.randn(1, tokens, 4, 4, generator=generator)
    return x, residual, pre, post, comb


def test_decode_and_prefill_with_keyword_interface_restore_exactly():
    blocks = [Block(), Block()]
    model = SimpleNamespace(layers=blocks)
    originals = [(block.hc_pre, block.hc_post) for block in blocks]
    with FusedHCDecodeFold(model, backend="torch") as controller:
        for tokens in (1, 3):
            x, residual, pre, post, comb = tensors(tokens)
            for block, (original_pre, original_post) in zip(blocks, originals):
                assert torch.equal(block.hc_pre(residual, pre_mix=pre), original_pre(residual, pre))
                assert torch.equal(block.hc_post(x, residual, post=post, comb=comb), original_post(x, residual, post, comb))
        assert controller.stats == {"pre_selected": 2, "post_selected": 2, "pre_passthrough": 2, "post_passthrough": 2}
        assert controller.describe()["per_block_coverage"] == [
            {"block": 0, "pre": 1, "post": 1}, {"block": 1, "pre": 1, "post": 1}]
    for block, (pre, post) in zip(blocks, originals):
        assert block.hc_pre.__func__ is pre.__func__
        assert block.hc_post.__func__ is post.__func__
        assert "hc_pre" not in block.__dict__ and "hc_post" not in block.__dict__
        assert not hasattr(block, "_ascend_decode_hc_fold_owner")


def test_existing_instance_override_is_preserved():
    block = Block()
    original = block.hc_pre
    override = lambda x, pre_mix: original(x, pre_mix)
    block.hc_pre = override
    controller = FusedHCDecodeFold(SimpleNamespace(layers=[block]), backend="torch")
    controller.install(); controller.close(); controller.close()
    assert block.hc_pre is override
    with pytest.raises(RuntimeError, match="fresh"):
        controller.install()


def test_foreign_owner_or_method_is_not_overwritten():
    block = Block()
    controller = FusedHCDecodeFold(SimpleNamespace(layers=[block]), backend="torch").install()
    with pytest.raises(RuntimeError, match="owner"):
        FusedHCDecodeFold(SimpleNamespace(layers=[block]), backend="torch")
    replacement = block.hc_pre
    foreign = lambda *_: None
    block.hc_pre = foreign
    with pytest.raises(RuntimeError, match="owner"):
        controller.close()
    assert block.hc_pre is foreign
    block.hc_pre = replacement
    controller.close()


def test_bad_input_does_not_change_original_model_interface():
    block = Block()
    with FusedHCDecodeFold(SimpleNamespace(layers=[block]), backend="torch"):
        _, residual, pre, _, _ = tensors()
        with pytest.raises(ValueError):
            block.hc_pre(residual, pre.bfloat16())
    assert "hc_pre" not in block.__dict__


def test_install_validates_every_block_before_modifying_any():
    blocks = [Block(), Block()]
    controller = FusedHCDecodeFold(SimpleNamespace(layers=blocks), backend="torch")
    foreign = lambda *_: None
    blocks[1].hc_post = foreign
    with pytest.raises(RuntimeError, match="owner"):
        controller.install()
    assert "hc_pre" not in blocks[0].__dict__
    assert not hasattr(blocks[0], "_ascend_decode_hc_fold_owner")
    assert blocks[1].hc_post is foreign


def test_empty_or_incompatible_block_is_refused():
    with pytest.raises(ValueError):
        FusedHCDecodeFold(SimpleNamespace(layers=[]), backend="torch")
    block = Block(); block.hc_mult = 2
    with pytest.raises(ValueError):
        FusedHCDecodeFold(SimpleNamespace(layers=[block]), backend="torch")
