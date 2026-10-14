"""The Ascend provider's GGUF raw-block arithmetic, against each format's own decoder.

`relicllm/components/gguf/ascend_ops.py` answers five of the GGUF raw-block bindings in plain torch
over two formats -- `iq4_nl`, which is the whole trunk of Xing4.0-29B-A4B, and `q6_k`, which is its
one `output.weight`.  Once the provider is registered, `relicllm/runtime/ops.load_ops("ascend")`
hands those bindings back the way it hands back the built CUDA extension on a CUDA box.  Nothing
here needs a 910: the provider is torch over whatever device the tensors are already on, so the
arithmetic is checkable on a host with no Ascend anything -- which is the point of testing it here,
because the alternative is discovering a wrong offset on the card.

The comparisons are aimed at the two seams where a torch re-statement of a raw-block kernel goes
wrong, rather than at fp32 accumulation:

* **The fold.**  `iq4_nl.fold_to_runtime_span` regroups eight native 18-byte blocks into one
  144-byte row element, and the decoder has to un-fold that.  Read the bytes at the wrong offset and
  the answer is a different number rather than a different rounding, so every GEMM here is compared
  against the reference decoder's own fp32 weights.
* **The Q6_K split.**  A Q6_K weight's six bits live in two places -- its low four in `ql` and its
  top two in a two-bit field of `qh` -- and the four sub-lanes read them in an order that is not the
  one a reshape gives you.  Get the pairing or the sub-scale lane wrong and the answer is a
  different number, so the Q6_K decode is compared against `tensor_reader._read_q6_k_rows` and the
  sub-scale layout is pinned by a row whose values are written down from the format's arithmetic.
* **The arithmetic the kernel does around the dot.**  The leading axis `gguf_quant_gemm_pair_forward`
  stacks on, the CSR walk that counts routes rather than tokens, `route_weights` applied *before* the
  down projection, and the swiglu clamp on the gate's positive tail only -- each is a decision the
  CUDA kernel made and the call sites depend on, and each is pinned against a reference written from
  the call site rather than from the provider.

What is compared is therefore the whole of the arithmetic, with one thing held fixed per device:
the reference the provider is measured against is built with the same multiply the card performs.
On the host that is an fp32 product; on a 910A the cube unit rounds both operands to fp16 before it
multiplies, so the reference rounds them the same way -- see `_gemm` for the measurement.  Loosening
a tolerance instead would be asserting a precision the card does not have and hiding the wrong
offset this module exists to catch.

The last tests read the released checkpoint's own tensors when it is present, because a
synthetic payload is uniform in ways a quantizer's output is not: put the file at
`$POCKETLLM_XING4_GGUF`, or a `Xing4.0-29B-A4B-GGUF/xing4_0-29b-IQ4_NL.gguf` under
`$POCKETLLM_XING4_DIR` (default `/mnt/data2`), and they run; without one they skip, and a skip is
not a pass.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from relicllm.loader.gguf import iq4_nl
from relicllm.loader.gguf.bundle import read_gguf_bundle
from relicllm.loader.gguf.quant_types import GGUF_DENSE_TYPE_IDS
from relicllm.loader.gguf.tensor_reader import GGUFTensorDataReader
from relicllm.components.gguf import ascend_ops
from relicllm.runtime import ops as ops_plane
from tests.gguf_test_utils import GGML_Q6_K, write_gguf


#: The two runtime type ids the provider decodes, read out of the loader's own dispatch table rather
#: than written here, so this file and `ascend_ops` cannot disagree about which number means what.
IQ4_NL_TYPE_ID = int(GGUF_DENSE_TYPE_IDS["iq4_nl"])
Q6_K_TYPE_ID = int(GGUF_DENSE_TYPE_IDS["q6_k"])

#: A Q6_K super-block: 256 weights in 210 bytes.
Q6_K_BLOCK_ELEMS = 256
Q6_K_BLOCK_BYTES = 210


GGUF_ENV = "POCKETLLM_XING4_GGUF"
CHECKPOINT_DIR_ENV = "POCKETLLM_XING4_DIR"
CHECKPOINT_DIR_DEFAULT = "/mnt/data2"
GGUF_NAME = "Xing4.0-29B-A4B-GGUF/xing4_0-29b-IQ4_NL.gguf"

#: Layer 2's routed gate stack, one 3584 x 1024 plane per expert -- the shape the MoE reads.
REAL_ROUTED_TENSOR = "blk.2.ffn_gate_exps.weight"

#: Layer 0 is a dense block, so its MLP is a plain 2-D matrix: 3584 in, 9216 out.
REAL_DENSE_TENSOR = "blk.0.ffn_gate.weight"

#: The vocabulary projection, and the checkpoint's one `q6_k` tensor: 3584 in, 152k vocab rows.
REAL_LM_HEAD_TENSOR = "output.weight"

#: The five bindings the provider answers, in the order `BINDINGS` declares them -- which is the
#: order `answered_bindings` promises, and not the order the functions are defined in.
ANSWERED = (
    "gguf_quant_gemm_forward",
    "gguf_quant_gemm_pair_forward",
    "gguf_quant_gemm_prefill_forward",
    "gguf_quant_embedding_forward",
    "gguf_moe_prefill_grouped_forward",
)

#: The one binding the provider is *decidedly* not answering: `models/xing4_0/hyper_connection.py`
#: already treats a missing op as "use the eager arithmetic", and that arithmetic is the same sum.
DELIBERATELY_ABSENT = "xing4_hyper_connection_forward"

#: The provider returns the fp32 accumulator where the CUDA kernels store bf16 and their callers
#: narrow again, so this tolerance is an accumulation order rather than a store width.  Row-relative,
#: because an element near zero in a row whose dot is O(200) carries the row's absolute error.
FP32_ROW_TOL = 1e-5

#: The routed MoE's nonlinear chain, on the card.  `_gemm` below rounds the reference the way the
#: cube rounds, so the residual here is not the operand rounding itself but what the swiglu adds on
#: top of it: the gate's activation re-rounds an operand that is already an fp16-rounded GEMM, and
#: the down projection then rounds that product again.  Measured on a first-generation 910 (CANN
#: 9.0.0, torch_npu 2.10.0): the provider's chunked chain against an fp16-operand reference is
#: 4.2e-4 at `swiglu_limit=0` and 3.2e-4 at 1.0, so the bound sits above that.  On the host the
#: chain is the same products summed in a different order and `FP32_ROW_TOL` still holds.
NPU_MOE_ROW_TOL = 1e-3


@pytest.fixture(autouse=True)
def clean_registry():
    """Take the provider registry out and put it back, so this module cannot leak a registration.

    Several tests below call `ascend_ops.register()`, which is a global mutation, and
    `test_ops_provider.py` asserts things about an empty registry on a platform with no provider.
    Two files that both reach into `_PROVIDERS` is what makes this fixture the honest version of
    "clean up after yourself".
    """
    before = dict(ops_plane._PROVIDERS)
    yield
    ops_plane._PROVIDERS.clear()
    ops_plane._PROVIDERS.update(before)


def _devices() -> tuple[str, ...]:
    """`cpu`, plus `npu` on a host where `torch_npu` is loaded and sees a card.

    The provider takes whatever device its tensors are on, so the same module runs on both; the npu
    leg is the one that says so on the hardware this is for.
    """
    devices = ["cpu"]
    try:
        import torch_npu  # noqa: F401
    except Exception:
        return tuple(devices)
    if torch.npu.is_available():
        devices.append("npu")
    return tuple(devices)


DEVICES = _devices()


def _provider():
    """What `load_ops("ascend")` hands back: the namespace, through the door, not the module.

    `register()` is called here rather than in a fixture because `clean_registry` takes the registry
    back after every test, so this module has to put the provider back each time it asks -- the same
    act `components/gguf/quantized_ops.py` performs on an Ascend host at import.
    """
    ascend_ops.register()
    ops = ops_plane.load_ops("ascend")
    assert ops is not None
    return ops


def _grid(device: str) -> torch.Tensor:
    """The empty int8 tensor the call sites pass as `signed_grid`.

    IQ4_NL reads its values from a 16-entry codebook the kernel holds and Q6_K has no codebook at
    all, so neither needs a grid; the argument is in the signature because the other formats' is.
    """
    return torch.empty(0, dtype=torch.int8, device=device)


def _marker_blocks(rows: int, row_elems: int, *, seed: int) -> np.ndarray:
    """Native 18-byte blocks with real codebook indices and real fp16 scales.

    Both nibbles of every byte are populated and the scales span a decade, so a decoder that reads
    only the low half, transposes the halves or takes the scale from the wrong byte lands on a
    different number rather than on the same one twice.
    """
    rng = np.random.default_rng(seed)
    n_blocks = rows * iq4_nl.blocks_per_row(row_elems)
    payload = np.empty((n_blocks, iq4_nl.IQ4_NL_BLOCK_BYTES), dtype=np.uint8)
    payload[:, :2] = np.frombuffer(
        rng.uniform(0.004, 0.05, size=n_blocks).astype("<f2").tobytes(), dtype=np.uint8
    ).reshape(n_blocks, 2)
    payload[:, 2:] = rng.integers(0, 256, size=(n_blocks, 16), dtype=np.uint8)
    return payload.reshape(rows, iq4_nl.blocks_per_row(row_elems), iq4_nl.IQ4_NL_BLOCK_BYTES)


def _folded(blocks: np.ndarray, row_elems: int, device: str) -> torch.Tensor:
    """The loader's geometry, on the device: eight native blocks per 144-byte row element."""
    folded = iq4_nl.fold_to_runtime_span(np.asarray(blocks), row_elems)
    return torch.as_tensor(folded.copy(), device=device)


def _decoded(blocks: np.ndarray, row_elems: int) -> np.ndarray:
    """`(rows, row_elems)` fp32 weights, out of the format's own decoder.

    Exact rather than approximate -- a codebook entry is a small integer and the scale is a finite
    fp16 -- so this is the format's arithmetic and the only difference left is accumulation order.
    """
    native = np.asarray(blocks)
    return iq4_nl.dequantize_blocks(native).reshape(native.shape[-3], int(row_elems))


def _gemm(x: np.ndarray, w: np.ndarray, device: str) -> np.ndarray:
    """`x @ w.T`, with the operands narrowed the way ``device`` narrows them before it multiplies.

    The decode is exact on both devices -- a codebook entry is a small integer and the scale is a
    finite fp16 -- so the only difference between the provider and the format's own decoder is the
    multiply, and this is the reference's half of it.  On the host that multiply is fp32, and the
    fp32 products are the format's arithmetic.  On a 910A it is *not*: measured on a first-generation
    910 (CANN 9.0.0, torch_npu 2.10.0) an ``npu`` fp32 ``matmul`` against an fp64 reference carries a
    row-relative error of 2.37e-4, flat across K = 256, 1024, 3584 and M = 1, 8, 64, while rounding
    both operands to fp16 reproduces the card's answer to 1.02e-6 -- and bf16 operands would be
    2.4e-3.  So the cube rounds fp32 operands to fp16 before it multiplies, no
    ``torch.npu.matmul.allow_hf32`` setting changes that, and a reference that keeps the fp32
    products would measure the card's precision rather than the provider's arithmetic.  Rounding the
    operands here instead leaves an accumulation order, which `FP32_ROW_TOL` covers.
    """
    if device == "npu":
        x = x.astype(np.float16).astype(np.float32)
        w = w.astype(np.float16).astype(np.float32)
    return x @ w.T


def _moe_tol(device: str) -> float:
    """`NPU_MOE_ROW_TOL` on the card, `FP32_ROW_TOL` on the host -- see :data:`NPU_MOE_ROW_TOL`."""
    return NPU_MOE_ROW_TOL if device == "npu" else FP32_ROW_TOL


def _assert_matches(got: np.ndarray, want: np.ndarray, tol: float = FP32_ROW_TOL) -> None:
    """Row-relative, so an element near zero in a wide row is not read as a mismatch."""
    scale = np.maximum(np.abs(np.asarray(want)).max(axis=-1, keepdims=True), 1e-6)
    err = np.abs(np.asarray(got) - np.asarray(want)) / scale
    assert float(err.max()) <= tol, (float(err.max()), tol)


def _sigmoid(a: np.ndarray) -> np.ndarray:
    """Stable logistic, so a large negative gate does not warn `exp` overflow on the way to zero."""
    out = np.empty_like(a, dtype=np.float32)
    positive = a >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-a[positive]))
    exp_a = np.exp(a[~positive])
    out[~positive] = exp_a / (1.0 + exp_a)
    return out


def _gguf_path() -> Path:
    override = os.environ.get(GGUF_ENV)
    if override:
        path = Path(override)
    else:
        root = Path(os.environ.get(CHECKPOINT_DIR_ENV, CHECKPOINT_DIR_DEFAULT))
        path = root / GGUF_NAME
    if not path.exists():
        pytest.skip(f"set {GGUF_ENV}, or place the GGUF at {path}, to run this")
    return path


# --------------------------------------------------------------------------- #
# The decoder: the fold, the nibble order, the codebook
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("row_elems", [256, 1024, 3584])
def test_the_decoder_reproduces_the_format_s_own_decoder(device: str, row_elems: int) -> None:
    """The whole of the fold claim: un-fold the 144 bytes to eight blocks, decode, compare.

    Every width the model uses is here, and 3584 is the one that is not a power of two -- so a
    decoder that derived its span count from a shift rather than from the tensor's own shape fails
    on it.
    """
    blocks = _marker_blocks(3, row_elems, seed=row_elems)
    got = ascend_ops._iq4_nl_weights(_folded(blocks, row_elems, device), row_elems, torch.float32)
    want = _decoded(blocks, row_elems)
    assert tuple(got.shape) == (3, row_elems)
    _assert_matches(got.cpu().numpy(), want)
    assert np.abs(want).max() > 1.0, "a reference that is all zeros would pass a relative check"


@pytest.mark.parametrize("device", DEVICES)
def test_the_low_and_high_nibbles_are_not_interleaved(device: str) -> None:
    """Byte `j` is weight `j` in its low nibble and weight `j + 16` in its high one.

    An alternating read is IQ4_NL's most plausible wrong answer: it is another real packing, it is
    what several 4-bit formats do, and it survives an all-ones row.  A row that counts makes it fail.
    """
    row_elems, rows = 256, 1
    blocks = np.zeros((rows, iq4_nl.blocks_per_row(row_elems), iq4_nl.IQ4_NL_BLOCK_BYTES), dtype=np.uint8)
    blocks[:, :, :2] = np.frombuffer(np.float16(1.0).tobytes(), dtype=np.uint8)
    for j in range(16):
        blocks[:, :, 2 + j] = j | (j << 4)
    got = ascend_ops._iq4_nl_weights(_folded(blocks, row_elems, device), row_elems, torch.float32)
    table = iq4_nl.kvalues().astype(np.float32)
    want = np.tile(np.concatenate([table, table]), row_elems // iq4_nl.QK_IQ4_NL)[None, :]
    _assert_matches(got.cpu().numpy(), want)


@pytest.mark.parametrize("device", DEVICES)
def test_one_codebook_index_everywhere_is_a_constant_row(device: str) -> None:
    """Index 7 on both nibbles with a unit scale: the row is `-10`, and the dot over ones is exact.

    A decoder that masks the wrong nibble, or takes the scale from the block's last two bytes, lands
    on a constant -- and a constant row times a plausible activation is still plausible.  Pinning
    the value is what makes the constant visible.
    """
    row_elems, rows = 256, 2
    blocks = np.zeros((rows, iq4_nl.blocks_per_row(row_elems), iq4_nl.IQ4_NL_BLOCK_BYTES), dtype=np.uint8)
    blocks[:, :, :2] = np.frombuffer(np.float16(1.0).tobytes(), dtype=np.uint8)
    blocks[:, :, 2:] = 0x77
    assert iq4_nl.kvalues()[7] == -10
    y = _provider().gguf_quant_gemm_forward(
        torch.ones(1, row_elems, device=device),
        _folded(blocks, row_elems, device),
        row_elems,
        iq4_nl.IQ4_NL_FILE_TYPE_ID,
        _grid(device),
    )
    assert y.dtype == torch.float32
    assert torch.equal(y.cpu(), torch.full((1, rows), -10.0 * row_elems))


def test_the_compute_width_follows_the_activation_and_falls_back_to_fp32() -> None:
    """The rule the decode is built on, and it is a choice the CUDA kernel also makes.

    Mirroring the activation's width keeps an fp16 GEMM on Ascend's cube unit; anything this cannot
    hold -- an int64 id tensor, a bool mask -- widens to fp32 rather than going through fp16.
    """
    assert ascend_ops._compute_dtype(torch.zeros(1, dtype=torch.float16)) is torch.float16
    assert ascend_ops._compute_dtype(torch.zeros(1, dtype=torch.bfloat16)) is torch.bfloat16
    assert ascend_ops._compute_dtype(torch.zeros(1, dtype=torch.float32)) is torch.float32
    assert ascend_ops._compute_dtype(torch.zeros(1, dtype=torch.int64)) is torch.float32


@pytest.mark.parametrize("device", DEVICES)
def test_the_chunked_decode_is_the_unchunked_decode(device: str, monkeypatch) -> None:
    """`_DECODE_CHUNK_BLOCKS` bounds a transient; it does not change the arithmetic.

    The chunk loop exists so a 152k-row vocabulary does not allocate two eight-byte temporaries per
    weight on the way to a two-byte one.  A boundary that dropped or double-counted a block would be
    invisible at the default 1<<20 -- which no test can afford to allocate -- so this drives the loop
    at a size where every boundary is crossed.
    """
    row_elems, rows = 512, 3
    folded = _folded(_marker_blocks(rows, row_elems, seed=99), row_elems, device)
    whole = ascend_ops._iq4_nl_weights(folded, row_elems, torch.float32)
    monkeypatch.setattr(ascend_ops, "_DECODE_CHUNK_BLOCKS", 2)
    chunked = ascend_ops._iq4_nl_weights(folded, row_elems, torch.float32)
    assert torch.equal(whole.cpu(), chunked.cpu())


# --------------------------------------------------------------------------- #
# The bindings the model calls
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("row_elems,rows_out", [(256, 5), (1024, 4), (3584, 3)])
def test_the_decode_gemm_is_the_reference_sum(device: str, row_elems: int, rows_out: int) -> None:
    """One row in, one row out, over a matrix with more rows than the input has."""
    blocks = _marker_blocks(rows_out, row_elems, seed=row_elems)
    x = np.random.default_rng(1).standard_normal((1, row_elems)).astype(np.float32)
    y = _provider().gguf_quant_gemm_forward(
        torch.as_tensor(x, device=device),
        _folded(blocks, row_elems, device),
        row_elems,
        iq4_nl.IQ4_NL_FILE_TYPE_ID,
        _grid(device),
    )
    assert tuple(y.shape) == (1, rows_out)
    _assert_matches(y.cpu().numpy(), _gemm(x, _decoded(blocks, row_elems), device))


@pytest.mark.parametrize("device", DEVICES)
def test_the_gemm_keeps_the_leading_axes_of_a_ragged_batch(device: str) -> None:
    """`[*x.shape[:-1], N]`, which is what a `[B, S, K]` caller reads.

    The CUDA kernel's output shape is `[*x.shape[:-1], N]` and the wrapper reshapes nothing, so a
    provider that returned `[B*S, N]` would pass a matmul check and break the caller.
    """
    row_elems, rows_out, batch, seq = 256, 3, 2, 5
    blocks = _marker_blocks(rows_out, row_elems, seed=3)
    x = np.random.default_rng(4).standard_normal((batch, seq, row_elems)).astype(np.float32)
    y = _provider().gguf_quant_gemm_prefill_forward(
        torch.as_tensor(x, device=device),
        _folded(blocks, row_elems, device),
        row_elems,
        iq4_nl.IQ4_NL_FILE_TYPE_ID,
        _grid(device),
    )
    assert tuple(y.shape) == (batch, seq, rows_out)
    want = _gemm(x.reshape(-1, row_elems), _decoded(blocks, row_elems), device)
    _assert_matches(y.cpu().numpy(), want.reshape(batch, seq, rows_out))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n_rows", [2, 3, 5, 8])
def test_the_prefill_binding_answers_the_reference_for_every_row_count(device: str, n_rows: int) -> None:
    """One implementation behind two names, which is what the provider claims they are.

    The CUDA side has two bindings because it tiles rows > 1 differently, and there is a real choice
    to make there.  Here the two are the same expression, so what is worth testing is that neither
    the name nor the number of rows riding along changes the answer.  The two names are compared
    within an accumulation order rather than bitwise: `[n_rows, K] @ [K, N]` and `[1, K] @ [K, N]`
    are two different GEMM kernels, and the last ulp of a reduction is theirs to choose.
    """
    row_elems, rows_out = 256, 3
    blocks = _marker_blocks(rows_out, row_elems, seed=n_rows)
    x = np.random.default_rng(n_rows).standard_normal((n_rows, row_elems)).astype(np.float32)
    provider = _provider()
    folded = _folded(blocks, row_elems, device)
    x_t = torch.as_tensor(x, device=device)
    prefill = provider.gguf_quant_gemm_prefill_forward(
        x_t, folded, row_elems, iq4_nl.IQ4_NL_FILE_TYPE_ID, _grid(device)
    )
    rows = [
        provider.gguf_quant_gemm_forward(
            x_t[row : row + 1], folded, row_elems, iq4_nl.IQ4_NL_FILE_TYPE_ID, _grid(device)
        )
        for row in range(n_rows)
    ]
    assert tuple(prefill.shape) == (n_rows, rows_out)
    want = _gemm(x, _decoded(blocks, row_elems), device)
    _assert_matches(prefill.cpu().numpy(), want)
    _assert_matches(torch.cat(rows, dim=0).cpu().numpy(), prefill.cpu().numpy())


@pytest.mark.parametrize("device", DEVICES)
def test_the_pair_binding_stacks_on_the_leading_axis(device: str) -> None:
    """`torch.stack({out0, out1}, 0)`, because the call site unpacks with `y0, y1 = ...`.

    Equal row counts, because that is the shape `QuantizedGGUFLinear.pair` accepts -- it refuses two
    weights of different output dims before it calls this -- and because `torch.stack` cannot answer
    otherwise.  The two weights are different matrices, so a swap is still visible in the comparison
    of each half.
    """
    row_elems = 512
    first_rows = second_rows = 5
    blocks0 = _marker_blocks(first_rows, row_elems, seed=11)
    blocks1 = _marker_blocks(second_rows, row_elems, seed=12)
    x = np.random.default_rng(13).standard_normal((4, row_elems)).astype(np.float32)
    y = _provider().gguf_quant_gemm_pair_forward(
        torch.as_tensor(x, device=device),
        _folded(blocks0, row_elems, device),
        row_elems,
        iq4_nl.IQ4_NL_FILE_TYPE_ID,
        _folded(blocks1, row_elems, device),
        row_elems,
        iq4_nl.IQ4_NL_FILE_TYPE_ID,
        _grid(device),
    )
    assert tuple(y.shape) == (2, 4, second_rows)
    want = np.stack(
        [
            _gemm(x, _decoded(blocks0, row_elems), device),
            _gemm(x, _decoded(blocks1, row_elems), device),
        ],
        axis=0,
    )
    _assert_matches(y.cpu().numpy(), want)


@pytest.mark.parametrize("device", DEVICES)
def test_the_embedding_binding_selects_rows_and_keeps_the_id_shape(device: str) -> None:
    """`[*ids.shape, row_elems]`, with the ids in any shape and any integer width.

    Xing4.0 does not reach this binding -- its `token_embd.weight` is dense and read through
    `read_dense` -- so this is the place a quantized embedding gets an answer instead of the CUDA
    kernel's q4_k/q5_k-only refusal.
    """
    row_elems, rows = 256, 9
    blocks = _marker_blocks(rows, row_elems, seed=21)
    ids = torch.tensor([[3, 0], [8, 3]], dtype=torch.int32, device=device)
    y = _provider().gguf_quant_embedding_forward(
        ids, _folded(blocks, row_elems, device), row_elems, iq4_nl.IQ4_NL_FILE_TYPE_ID, _grid(device)
    )
    assert tuple(y.shape) == (2, 2, row_elems)
    _assert_matches(y.cpu().numpy(), _decoded(blocks, row_elems)[ids.cpu().numpy()])


def _moe_routes(tokens: int, top_k: int, experts: int, picks: np.ndarray, weights: np.ndarray):
    """`seg_starts`, `route_tokens` and `route_weights` in the CSR form the kernel counts in."""
    seg = np.zeros(experts + 1, dtype=np.int32)
    for expert in range(experts):
        seg[expert + 1] = seg[expert] + int((picks == expert).sum())
    route_tokens = np.empty(seg[-1], dtype=np.int64)
    route_weights = np.empty(seg[-1], dtype=np.float32)
    cursor = seg[:-1].copy()
    for token in range(tokens):
        for slot in range(top_k):
            expert = int(picks[token, slot])
            route_tokens[cursor[expert]] = token
            route_weights[cursor[expert]] = weights[token, slot]
            cursor[expert] += 1
    return seg, route_tokens, route_weights


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("swiglu_limit", [0.0, 1.0])
def test_the_grouped_moe_binding_reproduces_the_per_expert_loop(device: str, swiglu_limit: float) -> None:
    """The routed MoE end to end, against a reference written the way the call site describes it.

    The kernel counts *routes*: `seg_starts` names the expert of each segment of the route axis and
    `route_tokens[r]` names the activation row route `r` reads.  So a provider that read tokens as
    routes -- the obvious misreading -- passes nothing here, and neither does one that dropped the
    `index_add_` or summed an expert's down projections without its route weights.

    The clamp is exercised rather than assumed: at `swiglu_limit=1.0` the activation is wide enough
    that the gate's positive tail and the up half both clip, which is what pins the clamp *before*
    the activation and on the gate's positive side only.
    """
    dim, inter, experts = 256, 512, 4
    w1 = np.stack([_marker_blocks(inter, dim, seed=e) for e in range(experts)])
    w3 = np.stack([_marker_blocks(inter, dim, seed=100 + e) for e in range(experts)])
    w2 = np.stack([_marker_blocks(dim, inter, seed=200 + e) for e in range(experts)])
    w1_dec = np.stack([_decoded(w1[e], dim) for e in range(experts)])
    w3_dec = np.stack([_decoded(w3[e], dim) for e in range(experts)])
    w2_dec = np.stack([_decoded(w2[e], inter) for e in range(experts)])

    tokens, top_k = 3, 2
    rng = np.random.default_rng(7)
    x = (rng.standard_normal((tokens, dim)) * (6.0 if swiglu_limit else 0.05)).astype(np.float32)
    picks = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    weights = np.array([[0.6, 0.4], [0.5, 0.5], [0.7, 0.3]], dtype=np.float32)
    seg, route_tokens, route_weights = _moe_routes(tokens, top_k, experts, picks, weights)

    y = _provider().gguf_moe_prefill_grouped_forward(
        torch.as_tensor(x, device=device),
        torch.as_tensor(route_tokens, device=device),
        torch.as_tensor(route_weights, device=device),
        torch.as_tensor(seg, device=device),
        _folded(w1, dim, device),
        _folded(w3, dim, device),
        _folded(w2, inter, device),
        dim,
        iq4_nl.IQ4_NL_FILE_TYPE_ID,
        dim,
        iq4_nl.IQ4_NL_FILE_TYPE_ID,
        inter,
        iq4_nl.IQ4_NL_FILE_TYPE_ID,
        _grid(device),
        swiglu_limit,
    )

    want = np.zeros((tokens, dim), dtype=np.float32)
    for token in range(tokens):
        for slot in range(top_k):
            expert = int(picks[token, slot])
            gate = _gemm(x[token], w1_dec[expert], device)
            up = _gemm(x[token], w3_dec[expert], device)
            if swiglu_limit > 0.0:
                gate = np.minimum(gate, swiglu_limit)
                up = np.clip(up, -swiglu_limit, swiglu_limit)
            # silu(gate) * up: the gate is one factor of its own activation, not a sigmoid alone.
            hidden = gate * _sigmoid(gate) * up
            # Before the down projection, which is where `route_swiglu_cast_kernel` applies it.
            want[token] += _gemm(hidden * weights[token, slot], w2_dec[expert], device)
    assert tuple(y.shape) == (tokens, dim)
    assert y.dtype == torch.float32
    _assert_matches(y.cpu().numpy(), want, tol=_moe_tol(device))
    assert np.abs(want).max() > 1.0, "a comparison of zeros against zeros would pass"


@pytest.mark.parametrize("device", DEVICES)
def test_the_grouped_moe_binding_reads_routes_and_not_tokens(device: str) -> None:
    """A route table where the two readings disagree, and a token that is routed nowhere.

    `route_tokens` here is not `[0..n_routes)`, so a provider that used the route index as the
    activation row -- correct only for the plan `plan_routes` builds from a single-token decode --
    returns a different answer.  Token 0 receives no route and has to stay exactly zero.
    """
    dim, inter, experts = 256, 256, 2
    w1 = np.stack([_marker_blocks(inter, dim, seed=e) for e in range(experts)])
    w3 = np.stack([_marker_blocks(inter, dim, seed=10 + e) for e in range(experts)])
    w2 = np.stack([_marker_blocks(dim, inter, seed=20 + e) for e in range(experts)])
    w1_dec = np.stack([_decoded(w1[e], dim) for e in range(experts)])
    w3_dec = np.stack([_decoded(w3[e], dim) for e in range(experts)])
    w2_dec = np.stack([_decoded(w2[e], inter) for e in range(experts)])
    x = np.random.default_rng(31).standard_normal((4, dim)).astype(np.float32)
    # Expert 0 serves tokens 3, 1 and 3 again; expert 1 serves token 2.
    route_tokens = np.array([3, 1, 3, 2], dtype=np.int64)
    seg = np.array([0, 3, 4], dtype=np.int32)
    route_weights = np.array([0.5, 0.25, 0.25, 1.0], dtype=np.float32)
    y = _provider().gguf_moe_prefill_grouped_forward(
        torch.as_tensor(x, device=device),
        torch.as_tensor(route_tokens, device=device),
        torch.as_tensor(route_weights, device=device),
        torch.as_tensor(seg, device=device),
        _folded(w1, dim, device),
        _folded(w3, dim, device),
        _folded(w2, inter, device),
        dim,
        iq4_nl.IQ4_NL_FILE_TYPE_ID,
        dim,
        iq4_nl.IQ4_NL_FILE_TYPE_ID,
        inter,
        iq4_nl.IQ4_NL_FILE_TYPE_ID,
        _grid(device),
        0.0,
    )
    want = np.zeros((4, dim), dtype=np.float32)
    for route, token in enumerate(route_tokens):
        expert = 0 if route < int(seg[1]) else 1
        gate = _gemm(x[token], w1_dec[expert], device)
        up = _gemm(x[token], w3_dec[expert], device)
        hidden = gate * _sigmoid(gate) * up
        want[token] += _gemm(hidden * route_weights[route], w2_dec[expert], device)
    _assert_matches(y.cpu().numpy(), want, tol=_moe_tol(device))
    assert np.array_equal(want[0], np.zeros(dim, dtype=np.float32))


# --------------------------------------------------------------------------- #
# Q6_K: the second format, and the one tensor that needs it
# --------------------------------------------------------------------------- #


def _q6_k_payload(rows: int, row_elems: int, *, seed: int) -> np.ndarray:
    """Native Q6_K super-blocks with real bit fields, real int8 sub-scales and real fp16 scales.

    Both `ql` and `qh` are populated and the sixteen sub-scales span a signed range, so a decoder
    that read a weight's two high bits from the wrong field, took a sub-scale from the wrong lane,
    or read the super-block scale as unsigned lands on a different number rather than on the same one
    twice.
    """
    assert row_elems % Q6_K_BLOCK_ELEMS == 0, "a Q6_K row is a whole number of 256-weight blocks"
    blocks_per_row = row_elems // Q6_K_BLOCK_ELEMS
    rng = np.random.default_rng(seed)
    payload = np.empty((rows, blocks_per_row, Q6_K_BLOCK_BYTES), dtype=np.uint8)
    payload[:, :, 0:128] = rng.integers(0, 256, size=(rows, blocks_per_row, 128), dtype=np.uint8)
    payload[:, :, 128:192] = rng.integers(0, 256, size=(rows, blocks_per_row, 64), dtype=np.uint8)
    # Signed on both sides and never zero, so a lane that read the wrong slot is a different number
    # rather than a coincidentally equal one.
    sub_scales = rng.integers(-40, 41, size=(rows, blocks_per_row, 16)).astype(np.int8)
    sub_scales[sub_scales == 0] = 1
    payload[:, :, 192:208] = sub_scales.view(np.uint8)
    scale_bytes = rng.uniform(0.004, 0.05, size=(rows, blocks_per_row)).astype("<f2").tobytes()
    payload[:, :, 208:210] = np.frombuffer(scale_bytes, dtype=np.uint8).reshape(rows, blocks_per_row, 2)
    return payload


def _q6_k_reference(tmp_path: Path, rows: int, row_elems: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Write the blocks as a one-tensor GGUF, then read the geometry and the reference decode back.

    `read_quantized_matrix_rows_reference` is the format's own decoder
    (`tensor_reader._read_q6_k_rows`), so this is the oracle the provider is measured against rather
    than a second statement of the same arithmetic written here.
    """
    blocks = _q6_k_payload(rows, row_elems, seed=seed)
    path = tmp_path / "q6k.gguf"
    write_gguf(
        path,
        tensors=[("t.weight", (row_elems, rows), GGML_Q6_K)],
        payloads={"t.weight": blocks.tobytes()},
    )
    with GGUFTensorDataReader(str(path)) as reader:
        native, type_name, read_elems = reader.read_quantized_matrix_blocks("t.weight")
        assert type_name == "q6_k"
        assert int(read_elems) == row_elems
        reference = reader.read_quantized_matrix_rows_reference("t.weight", 0, rows)
    assert tuple(native.shape) == blocks.shape
    assert np.array_equal(native.numpy(), blocks), "the reader moved a byte it should not have"
    return native.numpy(), reference.numpy()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("row_elems", [256, 512, 3584])
def test_the_q6_k_decoder_reproduces_the_format_s_own_decoder(tmp_path: Path, device: str, row_elems: int) -> None:
    """The whole of the Q6_K claim: split the six bits correctly, apply the right sub-scale, decode.

    The reference is the loader's own fp32 decoder, so this catches a wrong field, a wrong lane and a
    wrong byte order at once -- and 3584 is the model's own vocabulary width.
    """
    native, want = _q6_k_reference(tmp_path, 3, row_elems, seed=row_elems)
    got = ascend_ops._q6_k_weights(torch.as_tensor(native, device=device), row_elems, torch.float32)
    assert tuple(got.shape) == (3, row_elems)
    _assert_matches(got.cpu().numpy(), want)
    assert np.abs(want).max() > 1.0, "a reference that is all zeros would pass a relative check"


@pytest.mark.parametrize("device", DEVICES)
def test_the_q6_k_sub_lanes_read_their_scales_in_the_format_s_order(device: str) -> None:
    """The eight sub-scales a 128-weight half reads, and the order the four sub-lanes take them in.

    A Q6_K half's four sub-lanes read `sc[is+0]`, `sc[is+2]`, `sc[is+4]` and `sc[is+6]` where `is`
    is 0 for lanes 0-15 and 1 for lanes 16-31.  With every code zero -- both nibbles of `ql` and both
    bits of `qh` clear -- a weight is `-32 * scale`, and the sub-scale is the only thing that varies
    across the row.  A decoder that read sub-lane `n`'s scale from `sc[n]` writes a permuted row
    here, which is the failure a comparison against an equally-permuted reference would not catch.
    """
    row_elems, rows = 256, 1
    payload = np.zeros((rows, 1, Q6_K_BLOCK_BYTES), dtype=np.uint8)
    sub_scales = np.arange(1, 17, dtype=np.int8)
    payload[:, :, 192:208] = sub_scales.view(np.uint8)
    payload[:, :, 208:210] = np.frombuffer(np.float16(1.0).tobytes(), dtype=np.uint8)
    got = ascend_ops._q6_k_weights(torch.as_tensor(payload, device=device), row_elems, torch.float32)
    want = (-32.0 * np.repeat(sub_scales.astype(np.float32), 16))[None, :]
    _assert_matches(got.cpu().numpy(), want, tol=0.0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("row_elems,rows_out", [(256, 5), (3584, 3)])
def test_the_q6_k_gemm_is_the_reference_sum(tmp_path: Path, device: str, row_elems: int, rows_out: int) -> None:
    """The Q6_K decode through `gguf_quant_gemm_forward`, against the format's own fp32 weights."""
    native, want = _q6_k_reference(tmp_path, rows_out, row_elems, seed=row_elems)
    x = np.random.default_rng(1).standard_normal((1, row_elems)).astype(np.float32)
    y = _provider().gguf_quant_gemm_forward(
        torch.as_tensor(x, device=device),
        torch.as_tensor(native, device=device),
        row_elems,
        Q6_K_TYPE_ID,
        _grid(device),
    )
    assert tuple(y.shape) == (1, rows_out)
    _assert_matches(y.cpu().numpy(), _gemm(x, want, device))


@pytest.mark.parametrize("device", DEVICES)
def test_the_embedding_binding_answers_q6_k_too(tmp_path: Path, device: str) -> None:
    """A second binding, so the type id is shown to flow through the whole group and not one call."""
    row_elems, rows = 256, 9
    native, want = _q6_k_reference(tmp_path, rows, row_elems, seed=71)
    ids = torch.tensor([[3, 0], [8, 3]], dtype=torch.int32, device=device)
    y = _provider().gguf_quant_embedding_forward(
        ids, torch.as_tensor(native, device=device), row_elems, Q6_K_TYPE_ID, _grid(device)
    )
    assert tuple(y.shape) == (2, 2, row_elems)
    _assert_matches(y.cpu().numpy(), want[ids.cpu().numpy()])


def test_a_q6_k_row_that_is_not_the_format_s_block_or_is_too_short_is_refused() -> None:
    """144 bytes where 210 were expected, and a row the super-blocks do not cover.

    The first is an IQ4_NL row element handed to the other decoder; the second is a caller that read
    the geometry from the wrong axis and would otherwise decode the next row's weights into this one.
    """
    with pytest.raises(ValueError) as excinfo:
        ascend_ops._q6_k_weights(torch.zeros(2, 1, 144, dtype=torch.uint8), 256, torch.float32)
    assert "210" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        ascend_ops._q6_k_weights(torch.zeros(2, 1, 210, dtype=torch.uint8), 512, torch.float32)
    assert "512" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# The refusals, which are the difference between an error and a wrong number
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("type_id", [0, 2, 5, 6, 12, 23, 29])
def test_a_type_id_this_cannot_decode_is_refused_by_name(type_id: int) -> None:
    """Every other runtime type raises rather than decoding as one of the two formats.

    The failure this prevents is the quiet one: a block read at another format's stride produces a
    number, and a wrong number in a MoE layer or a vocabulary projection is a wrong model.  These are
    *runtime* ids -- `GGUF_DENSE_TYPE_IDS` values -- because the provider switches on those and not on
    the GGML file ids, and the message has to name the one that was asked for.
    """
    with pytest.raises(NotImplementedError) as excinfo:
        ascend_ops._require_supported(type_id, "gguf_quant_gemm")
    assert str(type_id) in str(excinfo.value)
    assert "iq4_nl" in str(excinfo.value)
    assert "q6_k" in str(excinfo.value)


def test_both_decoded_type_ids_are_accepted_and_the_message_names_them() -> None:
    """The two the provider does answer: no raise, and a refusal that names both.

    `8` is the id that used to be refused -- it is `q8_0`'s GGML *file* id but `q6_k`'s runtime id,
    and the loader's dispatch table is the one the kernel switches on, so the provider has to accept
    it.  Pinning it here is what keeps the two tables from being read as one.
    """
    assert Q6_K_TYPE_ID == 8 and IQ4_NL_TYPE_ID == 20
    for type_id in (IQ4_NL_TYPE_ID, Q6_K_TYPE_ID):
        ascend_ops._require_supported(type_id, "gguf_quant_gemm")  # does not raise
    with pytest.raises(NotImplementedError) as excinfo:
        ascend_ops._require_supported(6, "gguf_quant_gemm")
    assert "iq4_nl" in str(excinfo.value) and "q6_k" in str(excinfo.value)


def test_each_binding_refuses_a_type_id_it_cannot_decode() -> None:
    """Through the bindings themselves: the guard is on the path the model takes, not beside it."""
    blocks = torch.zeros(2, 1, 144, dtype=torch.uint8)
    x = torch.ones(1, 256)
    with pytest.raises(NotImplementedError):
        _provider().gguf_quant_gemm_forward(x, blocks, 256, 6, _grid("cpu"))
    with pytest.raises(NotImplementedError):
        _provider().gguf_quant_embedding_forward(
            torch.zeros(1, dtype=torch.long), blocks, 256, 12, _grid("cpu")
        )
    with pytest.raises(NotImplementedError):
        _provider().gguf_moe_prefill_grouped_forward(
            x,
            torch.zeros(1, dtype=torch.long),
            torch.ones(1),
            torch.tensor([0, 1], dtype=torch.int32),
            blocks,
            blocks,
            blocks,
            256,
            6,
            256,
            20,
            256,
            20,
            _grid("cpu"),
            0.0,
        )


def test_a_row_element_that_is_not_the_loader_s_fold_is_refused() -> None:
    """18 bytes where 144 were expected, and a row the blocks do not cover.

    Both are refusals rather than reshapes: bytes that did not come through `fold_to_runtime_span`
    are not what this decodes, and a row longer than its blocks is a caller that read the geometry
    from the wrong axis -- which would otherwise decode the next row's weights into this one.
    """
    with pytest.raises(ValueError) as excinfo:
        ascend_ops._iq4_nl_weights(torch.zeros(2, 8, 18, dtype=torch.uint8), 256, torch.float32)
    assert "144" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        ascend_ops._iq4_nl_weights(torch.zeros(2, 1, 144, dtype=torch.uint8), 512, torch.float32)
    assert "512" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# The seam: what `load_ops("ascend")` hands back, and what it still does not
# --------------------------------------------------------------------------- #


def test_the_provider_answers_the_five_gguf_bindings_through_load_ops() -> None:
    """The door, not the module: `load_ops("ascend")` is the namespace `ascend_ops.ops()` builds."""
    ascend_ops.register()
    ops = ops_plane.load_ops("ascend")
    assert ops is not None
    assert ops is ascend_ops.ops()
    assert ops_plane.load_ops("ascend") is ops, "the namespace is cached rather than rebuilt"
    assert ascend_ops.answered_bindings() == ANSWERED
    for name in ANSWERED:
        assert callable(getattr(ops, name))


def test_the_gguf_component_registers_the_provider_on_the_host_that_needs_it() -> None:
    """Who registers it, and on which hosts: the module that reaches for the bindings, and only there.

    Importing `components/gguf/quantized_ops` is the act, which is why this test does it explicitly
    rather than trusting that an earlier one did.  The host is probed instead of assumed, so the
    assertion means the same thing on the box this was written on and on the one it is for: on a 910
    the five bindings resolve, and on a CUDA box `load_ops("ascend")` goes on answering `None`,
    because a provider for another platform is not something this host has.
    """
    from relicllm.components.gguf import quantized_ops  # noqa: F401  importing it is the registration
    from relicllm.runtime.device import probe_accelerator

    if probe_accelerator().platform == "ascend":
        assert ops_plane.load_ops("ascend") is ascend_ops.ops()
    else:
        assert ops_plane.load_ops("ascend") is None


def test_what_the_provider_does_not_answer_is_reported_rather_than_hidden() -> None:
    """`missing_bindings` is the honest half, and the hyper-connection is the one deliberate absence.

    Xing4.0's hyper-connection falls back to the eager arithmetic at its call site, which is the same
    sum -- so it must show up as missing rather than as a stub that silently returned `None`.  The
    rest of what is missing belongs to the other families.
    """
    missing = ops_plane.missing_bindings(ascend_ops.ops())
    assert DELIBERATELY_ABSENT in missing
    assert set(ANSWERED).isdisjoint(set(missing))
    assert len(missing) == len(ops_plane.BINDINGS) - len(ANSWERED)
    assert missing == tuple(name for name in ops_plane.BINDINGS if name not in set(ANSWERED))


def test_registering_the_provider_twice_is_the_same_as_registering_it_once() -> None:
    """Idempotent, which the import path needs: a launch may import the component after a test did."""
    ascend_ops.register()
    first = ops_plane.load_ops("ascend")
    ascend_ops.register()
    ascend_ops.register()
    assert ops_plane.load_ops("ascend") is first
    assert "ascend" in ops_plane.registered_platforms()


def test_a_platform_with_nothing_built_for_it_still_answers_none() -> None:
    """The branch `load_ops` was written for, asserted on a name nothing registers.

    `mps` is the stand-in, and it is also the shape Ascend has on a CUDA host: the provider is
    registered by the GGUF component on the platform that needs it, so a box that is not one answers
    `None` here exactly as it did before the provider existed.
    """
    assert ops_plane.load_ops("mps") is None
    assert ops_plane.load_ops("cpu") is None


# --------------------------------------------------------------------------- #
# The call-site wrappers, over a synthesized quantized tensor
# --------------------------------------------------------------------------- #


def _quantized_tensor(blocks: np.ndarray, row_elems: int, out_dim: int):
    """A `QuantizedGGUFTensor` in the loader's geometry, folded and on the host."""
    from relicllm.loader.gguf.quantized_tensor import QuantizedGGUFTensor

    return QuantizedGGUFTensor(
        source_name="synthetic",
        blocks=_folded(blocks, row_elems, "cpu"),
        type_name="iq4_nl",
        type_id=iq4_nl.IQ4_NL_FILE_TYPE_ID,
        row_elems=row_elems,
        out_dim=out_dim,
    )


def test_the_linear_wrapper_runs_on_the_provider(monkeypatch) -> None:
    """`QuantizedGGUFLinear`, `Embedding` and `pair`, which is how the model reaches all of this.

    These wrappers raise when `load_ops()` answers `None`, and they are what fixes the argument order
    and the prefill/decode split -- so a provider that is only correct when called by hand is not yet
    a provider.  `load_ops` is patched rather than the host made to be Ascend: the host platform is a
    property of the box, and this test is about a seam that has nothing to do with it.
    """
    from relicllm.components.gguf import quantized_ops

    provider = _provider()
    monkeypatch.setattr(quantized_ops, "load_ops", lambda *args, **kwargs: provider)

    row_elems, rows_out = 512, 3
    blocks = _marker_blocks(rows_out, row_elems, seed=41)
    linear = quantized_ops.QuantizedGGUFLinear(
        _quantized_tensor(blocks, row_elems, rows_out), out_dtype=torch.float32
    )
    weights = _decoded(blocks, row_elems)
    for n_rows in (1, 2):
        x = np.random.default_rng(n_rows).standard_normal((n_rows, row_elems)).astype(np.float32)
        _assert_matches(linear(torch.as_tensor(x)).numpy(), x @ weights.T)

    second_blocks = _marker_blocks(rows_out, row_elems, seed=42)
    second = quantized_ops.QuantizedGGUFLinear(
        _quantized_tensor(second_blocks, row_elems, rows_out), out_dtype=torch.float32
    )
    x = np.random.default_rng(43).standard_normal((2, row_elems)).astype(np.float32)
    y0, y1 = quantized_ops.QuantizedGGUFLinear.pair(linear, second, torch.as_tensor(x))
    _assert_matches(y0.numpy(), x @ weights.T)
    _assert_matches(y1.numpy(), x @ _decoded(second_blocks, row_elems).T)

    embedding = quantized_ops.QuantizedGGUFEmbedding(
        _quantized_tensor(blocks, row_elems, rows_out), out_dtype=torch.float32
    )
    ids = torch.tensor([2, 0, 2], dtype=torch.long)
    _assert_matches(embedding(ids).numpy(), weights[[2, 0, 2]])


def test_the_linear_wrapper_runs_a_q6_k_lm_head_on_the_provider(tmp_path: Path, monkeypatch) -> None:
    """`QuantizedGGUFLinear` over a `q6_k` weight, which is how the model reads `output.weight`.

    The lm_head is `QuantizedGGUFLinear(loader.read_quant("output.weight"), out_dtype=torch.float32)`,
    and `read_quant` hands back a `QuantizedGGUFTensor` whose `type_name` is `q6_k` -- so this is the
    call-site path a provider that only answered `iq4_nl` would fail on, at the last layer of every
    forward.  `load_ops` is patched for the same reason as above: the host platform is not the seam
    under test.
    """
    from relicllm.components.gguf import quantized_ops
    from relicllm.loader.gguf.quantized_tensor import QuantizedGGUFTensor

    provider = _provider()
    monkeypatch.setattr(quantized_ops, "load_ops", lambda *args, **kwargs: provider)

    row_elems, rows_out = 512, 3
    native, want = _q6_k_reference(tmp_path, rows_out, row_elems, seed=61)
    tensor = QuantizedGGUFTensor(
        source_name="synthetic",
        blocks=torch.as_tensor(native),
        type_name="q6_k",
        type_id=Q6_K_TYPE_ID,
        row_elems=row_elems,
        out_dim=rows_out,
    )
    linear = quantized_ops.QuantizedGGUFLinear(tensor, out_dtype=torch.float32)
    for n_rows in (1, 2):
        x = np.random.default_rng(n_rows).standard_normal((n_rows, row_elems)).astype(np.float32)
        _assert_matches(linear(torch.as_tensor(x)).numpy(), x @ want.T)


# --------------------------------------------------------------------------- #
# The released checkpoint's own tensors
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("device", DEVICES)
def test_a_real_routed_expert_matches_its_own_decode(device: str) -> None:
    """The released experts, whole-block, against the format's decoder.

    A synthetic payload can be uniform in ways a quantizer's output is not -- every scale inside one
    decade, every index populated -- so this is where a decoder that reads the scale from the wrong
    byte stops passing.
    """
    path = _gguf_path()
    expert_count = 2
    bundle = read_gguf_bundle(path)
    reader = GGUFTensorDataReader(bundle.tensors_by_name[REAL_ROUTED_TENSOR].shard_path)
    try:
        blocks = np.stack(
            [
                reader.read_routed_expert_blocks(REAL_ROUTED_TENSOR, expert)[0].numpy()
                for expert in range(expert_count)
            ]
        )
        _, type_name, row_elems = reader.read_routed_expert_blocks(REAL_ROUTED_TENSOR, 0)
    finally:
        reader.close()
    assert type_name == "iq4_nl"
    row_elems = int(row_elems)
    assert row_elems == 3584
    assert blocks.shape == (expert_count, 1024, 3584 // iq4_nl.QK_IQ4_NL, iq4_nl.IQ4_NL_BLOCK_BYTES)
    assert not np.array_equal(blocks[0], blocks[1]), "the reader ignored the expert index"
    x = np.random.default_rng(51).standard_normal((1, row_elems)).astype(np.float32)
    for expert in range(expert_count):
        y = _provider().gguf_quant_gemm_forward(
            torch.as_tensor(x, device=device),
            _folded(blocks[expert], row_elems, device),
            row_elems,
            iq4_nl.IQ4_NL_FILE_TYPE_ID,
            _grid(device),
        )
        want = _gemm(x, _decoded(blocks[expert], row_elems), device)
        _assert_matches(y.cpu().numpy(), want)
        assert np.abs(want).max() > 1.0


@pytest.mark.parametrize("device", DEVICES)
def test_a_real_dense_tensor_matches_its_own_decode(device: str) -> None:
    """The same format through `read_quantized_matrix_blocks`: the path a 2-D weight takes."""
    path = _gguf_path()
    bundle = read_gguf_bundle(path)
    reader = GGUFTensorDataReader(bundle.tensors_by_name[REAL_DENSE_TENSOR].shard_path)
    try:
        native, type_name, row_elems = reader.read_quantized_matrix_blocks(REAL_DENSE_TENSOR)
    finally:
        reader.close()
    assert type_name == "iq4_nl"
    blocks = native.numpy()
    row_elems = int(row_elems)
    assert blocks.shape[-1] == iq4_nl.IQ4_NL_BLOCK_BYTES
    x = np.random.default_rng(52).standard_normal((1, row_elems)).astype(np.float32)
    y = _provider().gguf_quant_gemm_forward(
        torch.as_tensor(x, device=device),
        _folded(blocks, row_elems, device),
        row_elems,
        iq4_nl.IQ4_NL_FILE_TYPE_ID,
        _grid(device),
    )
    assert tuple(y.shape) == (1, blocks.shape[-3])
    _assert_matches(y.cpu().numpy(), _gemm(x, _decoded(blocks, row_elems), device))


@pytest.mark.parametrize("device", DEVICES)
def test_the_real_lm_head_matches_its_own_decode(device: str) -> None:
    """The checkpoint's one `q6_k` tensor, `output.weight`, read the way the model reads it.

    The trunk being all `iq4_nl` is what makes the second decoder look optional; this is the tensor
    that proves it is not.  It is a row slice rather than the whole matrix because 152k rows of 210
    bytes is 447 MB of blocks the test does not need, and a two-row slice still crosses the
    super-block boundary a wrong lane or a wrong field would fail on.
    """
    path = _gguf_path()
    bundle = read_gguf_bundle(path)
    reader = GGUFTensorDataReader(bundle.tensors_by_name[REAL_LM_HEAD_TENSOR].shard_path)
    try:
        native, type_name, row_elems = reader.read_quantized_matrix_block_rows(REAL_LM_HEAD_TENSOR, 0, 2)
        reference = reader.read_quantized_matrix_rows_reference(REAL_LM_HEAD_TENSOR, 0, 2)
    finally:
        reader.close()
    assert type_name == "q6_k"
    blocks = native.numpy()
    row_elems = int(row_elems)
    assert row_elems == 3584
    assert blocks.shape == (2, row_elems // Q6_K_BLOCK_ELEMS, Q6_K_BLOCK_BYTES)
    assert not np.array_equal(blocks[0], blocks[1]), "the reader ignored the row slice"
    x = np.random.default_rng(53).standard_normal((1, row_elems)).astype(np.float32)
    y = _provider().gguf_quant_gemm_forward(
        torch.as_tensor(x, device=device),
        torch.as_tensor(blocks, device=device),
        row_elems,
        Q6_K_TYPE_ID,
        _grid(device),
    )
    assert tuple(y.shape) == (1, 2)
    _assert_matches(y.cpu().numpy(), _gemm(x, reference.numpy(), device))
    assert np.abs(reference.numpy()).max() > 1.0
