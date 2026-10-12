"""The precondition the real `DecodeGraphs` checks, and the failure it exists to prevent.

`tests/test_models_deepseek_v4_1_generate.py` fakes this class: the loop's contract with it -- built
once, installed on the model, released if the loop is left early -- needs no card, so the fake is what
the loop is driven against there. What a fake cannot stand in for is the class's own precondition,
because it is a fact about the *tree* rather than about the loop.

A CUDA graph records device work. A tree whose attention caches are in host memory does none of it
inside the capture, so what `capture_pass` records is an *empty* graph: the replay returns having run
none of the step, every reader of the sink goes on reading what the capture pass left there, and the
loop emits its first token over and over. That is a wrong answer carrying the flag of a fast one, and
it is what `--decode-graphs` used to mean on a run with no card in it. `v41_backend` and
`cli/generate_v41` answer it by resolving the option against where the tree actually went and asking
for the eager loop; this class answers it by refusing rather than recording nothing.

The refusal sits before `torch.cuda.graph_pool_handle`, so a host tree is refused on a host with no
card at all -- which is why the three cases below run anywhere. The third is the half a placement
check is easy to get wrong from the other side, over-refusing a tree that *is* on a card, and it is
the one that needs the card it asserts about.
"""

from __future__ import annotations

import pytest
import torch

from relicllm.models.deepseek_v4_1.graphs import DecodeGraphs, _cache_device

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="a capture needs a card")

WINDOW = 4


class _Attention(torch.nn.Module):
    """The buffer and the one other attribute the class reads: the placement off the cache, the
    compressor variant off the attention (``None`` is the no-compressor layer, ratio 0)."""

    def __init__(self, device: str, *, cached: bool = True) -> None:
        super().__init__()
        self.compressor = None
        if cached:
            self.register_buffer("window_kv_cache", torch.zeros(1, WINDOW, device=device))


class _Block(torch.nn.Module):
    def __init__(self, device: str, *, cached: bool = True) -> None:
        super().__init__()
        self.layer_id = 0
        self.attn = _Attention(device, cached=cached)


class _Backbone(torch.nn.Module):
    """A tree with the two attributes the class touches: the layers it walks and the caches it reads
    the placement off. The cache is registered on the attention module inside a block, at the depth
    the real tree registers it, so a reader that looked at the root would find nothing here either.
    """

    def __init__(self, device: str = "cpu", *, cached: bool = True, blocks: int = 2) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_Block(device, cached=cached) for _ in range(blocks)])


def test_a_host_resident_tree_is_refused_by_placement_and_told_why() -> None:
    """The run the option asks for and no card can serve. Both halves of the message are load-bearing:
    the placement is what the caller has to change -- the same tree on a card is a run this class can
    record -- and the empty graph is why the answer is a refusal rather than a slower run."""
    with pytest.raises(RuntimeError) as refusal:
        DecodeGraphs(_Backbone("cpu"))

    message = str(refusal.value)
    assert "are on cpu" in message
    assert "empty graph" in message


def test_a_model_without_the_cache_is_refused_by_name() -> None:
    """A different failure with its own answer: the class decides placement off a buffer, so a model
    that has none is not a tree on the host but not this model at all, and the message has to say
    which buffer it looked for rather than blaming the placement."""
    with pytest.raises(RuntimeError, match="window_kv_cache"):
        DecodeGraphs(_Backbone("cpu", cached=False))


def test_the_placement_is_read_off_the_buffer_itself() -> None:
    """The device in the refusal is the caller's only pointer to where the tree went, so the reader
    has to answer about the object the kernels write. The second block is the one asked about, so a
    walk that stopped at the first would not pass this by accident."""
    model = _Backbone("cpu")

    assert _cache_device(model) == model.layers[1].attn.window_kv_cache.device


@CUDA
def test_a_tree_whose_caches_are_on_a_card_is_not_refused() -> None:
    """The other half of the same predicate, and the half a check like this is easy to get wrong from
    the far side: a tree that *is* on a card has to reach the walk, because the walk is what a host
    tree loses -- one `LayerGraphs` installed per block, and the pool every recording will share.

    So the constructor is run with the real selection rather than `layer_ids=[]`, which would leave
    `layers` empty no matter what the walk did, and what is asserted is the installation. `captured`
    is still False afterwards because a constructor records nothing -- `capture_pass` is what
    records, and a layer recording a graph is `test_models_deepseek_v4_1_attention.py`'s subject.
    """
    model = _Backbone("cuda")
    graphs = DecodeGraphs(model)
    try:
        assert graphs.pool is not None
        assert graphs.captured is False
        assert [layer.block for layer in graphs.layers] == list(model.layers)
        assert all(block.decode_graph is layer for block, layer in zip(model.layers, graphs.layers))
    finally:
        graphs.release()
    # Releasing is the other end of installing: the blocks go back to the eager body, so a layer that
    # outlives the graphs does not replay into a pool nothing owns.
    assert all(block.decode_graph is None for block in model.layers)