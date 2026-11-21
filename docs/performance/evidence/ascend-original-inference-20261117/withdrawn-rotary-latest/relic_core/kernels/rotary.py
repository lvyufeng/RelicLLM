"""Explicit-arithmetic pair rotation for rotary position embeddings.

The reference rotates query and key vectors as ``complex64`` numbers: consecutive
elements of the last axis are one complex value and the position table supplies
``cos + i sin``. Written as source, that is ``(a*c - b*s, a*s + b*c)``; on a
compiled kernel each component may fuse one product into the add and round the
other on its own, which moves the result by about one FP32 ulp.

**The fused spellings are not universal, and this operator does not claim one.**
Which product fuses is a property of the reference build -- its compiler, target
architecture and math library -- and it can differ *per component*. Measured on
the Ascend host's aarch64 ATen, the real component matched
``real = fma(-b, s, round_f32(a*c))`` (the **second** product fused) on every one
of 4096 random pairs, while the imaginary component matched none of ``first``,
``second`` or the fully unfused form. On the x86 build the first-product form was
measured. A single carried-over formula therefore cannot be "the reference".

What this module does instead is state its arithmetic **explicitly** and measure
any host it is pointed at:

    DOCUMENTED_POLICY = ("first", "first")
    real = fma(a, c, -round_f32(b * s))
    imag = fma(a, s, +round_f32(b * c))

The fused step is ``tl.fma`` on device, never a compiler contraction, because
*which* product fuses is the contract. This policy is an **unbound experimental
operator arithmetic**: it is validated against an independent exact-rational
``fmaf`` oracle (``tests/test_ascend_rotary_math.py``, and the device oracle in
``tests/test_ascend_rotary.py``), and against a host's ATen complex multiply only
when ``reference_contraction()`` reports that host implements the same policy for
both components. On a host whose measurement differs, the two disagree by roughly
an ulp and the fused kernel's result is *not* claimed to equal that host's ATen;
the ``"torch"`` backend still matches it by construction, because that backend
*is* ATen. No backend selects its arithmetic from the measurement: switching
spellings on the fly would make a kernel's numbers depend on the machine that
launched it, so the policy is fixed in source and the measurement only reports.

The op is **in place** and writes back through the caller's strides, so a head
slice of a fused QKV buffer rotates without a copy::

    q  # [batch, sequence, heads, width], a slice of the fused buffer
    apply_rotary_inplace(q, table)                  # forward positions
    apply_rotary_inplace(k, table, inverse=True)    # reverse direction

The rank-3 form is ``[batch, sequence, width]`` -- the model's ``[b, s, d]`` with a
single implicit head, which is why the sequence axis is dim 1 there and not dim 0.
A decode step is therefore ``[b, 1, d]`` with a one-row table.

``table`` is FP32 ``[sequence, width // 2, 2]`` with ``(cos, sin)`` adjacent in
the last axis, or a single ``[width // 2, 2]`` row when the sequence axis is one
long (a decode step). It is built by the caller from its own positions; this
module never recomputes an angle, never builds an expanded per-element table and
never reads a "cos/sin separated" layout.

What this is not: it is not a new RoPE parameterisation and it does not change
any model's frequency schedule. It is the arithmetic of the multiply the
reference already performs, restated so that it runs on the NPU without a
complex64 kernel or an FP64 detour. Non-finite inputs are outside the reference's
finite contract: they propagate through IEEE arithmetic here, but their NaN
payloads are not a specified output.
"""
from __future__ import annotations

import importlib
import math

import torch

__all__ = ["DOCUMENTED_POLICY", "SPELLINGS", "apply_rotary_inplace",
           "matches_documented_policy", "reference_contraction", "supports_triton_ascend"]

#: Ascend's grid dimension cap, so a long sequence is covered by looping rows
#: inside each program rather than by asking for more programs than exist.
MAX_PROGRAMS = 65535
#: Lanes one program handles per width step. 1024 is the size the fused unpack
#: kernels already run at on this backend; it keeps the vector small enough for
#: the widest head while covering a narrow one in a single step.
K_TILE_MAX = 1024


def supports_triton_ascend() -> bool:
    """Report an installed Ascend Triton backend without starting device work."""
    try:
        module = importlib.import_module("relic_core.kernels.ascend_unpack")
    except ImportError:
        return False
    return bool(module.supports_triton_ascend())


#: Probe points where the three possible complex-multiply spellings disagree. Each
#: was selected so that first-product fusion, second-product fusion and the fully
#: unfused form give three *pairwise different* bit patterns -- a probe on which two
#: spellings coincide cannot identify the spelling, which is why these are searched
#: for rather than picked by hand. Fixed constants keep the probe free of any RNG
#: and identical on every host. The magnitudes are deliberately varied: the search
#: had to look beyond ordinary RoPE values, where the two fusions usually agree.
_CONTRACTION_PROBES = (
    (6.972659860071724e19, -1.44806181339136e14, -5.155183463272095e-13, -6.330575388346915e-07),
    (-1.474967655095139e28, -2.85523251822592e14, -9.069871520495099e-10, 1.2025029102858102e-20),
    (1.0341775213973392e-09, -0.023910559713840485, -1.1106357305115935e18, 5043744.0),
    (-1.779102524562022e20, -4.1380035345160524e23, -1.3305071020625059e-12, -1.8058788757575986e-17),
    (1.77150002024333e-13, 5.526848106103701e37, -4.4418546093843877e-32, 4.1566611709013126e21),
    (1.4845782995443166e34, 1.1934674120520636e21, -3.925852905448871e-21, 2.788750565407002e-26),
    (5.858836089064309e29, 14655000576.0, -1.6377054131684452e-24, 2.1598211219497898e-07),
    (-8.91819854763229e28, -5.523922920227051, -3.2588480636984947e-21, -9748.8173828125),
    # Additional fixed witnesses that separate ``second`` from ``unfused`` on the
    # real component (two of the eight above do not), and keep all three spellings
    # pairwise distinct on the imaginary component as well. Found by search over
    # random FP32 operands with the exponents spread wide, then frozen as constants
    # so the probe is identical on every host and free of any RNG.
    (-139412848.0, -113279168.0, 0.591978132724762, 0.8686354160308838),
    (-560944906240.0, 1009955110912.0, 0.00170822290237993, -0.0009643225348554552),
    (-36.266510009765625, -112.88300323486328, 2575173.5, -6345233.5),
    (59100246016.0, 221608771584.0, 0.0002724331570789218, -0.0008127092733047903),
)


def _complex_bits(pairs: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """ATen's complex multiply on ``[n, 2]`` FP32 pairs and ``[n, 2]`` angles."""
    product = (torch.view_as_complex(pairs.contiguous())
               * torch.view_as_complex(angles.contiguous()))
    return torch.view_as_real(product).reshape(-1, 2)


#: The arithmetic the fused kernel implements, as ``(real, imag)`` spellings.
#:
#: ``"first"`` fuses the pair's own first product, ``"second"`` fuses the other,
#: and ``"unfused"`` rounds both products separately. ``tests/
#: test_ascend_rotary_math.py`` re-derives both lines from exact rationals; the
#: measurement below only reports what a host does, it never selects this.
DOCUMENTED_POLICY = ("first", "first")

#: The spellings this module can name. A host that matches none of them is
#: reported as ``"unknown"`` for that component rather than coerced to a guess.
SPELLINGS = ("first", "second", "unfused")


def _candidate_component(pairs: torch.Tensor, angles: torch.Tensor, spelling: str,
                         component: int) -> torch.Tensor:
    """One component of ``(a*c - b*s, a*s + b*c)`` under a named spelling, as FP32.

    Component 0 is ``a*c - b*s`` and component 1 is ``a*s + b*c``. The product the
    spelling fuses into the add is formed exactly in FP64 (the operand pair is
    FP32, so a product is exact there); the one it does not fuse is rounded to
    FP32 first and then added, which is what a non-fused multiply contributes.
    """
    a = pairs[:, 0]
    b = pairs[:, 1]
    cosine = angles[:, 0]
    sine = angles[:, 1]
    wide_a, wide_b = a.to(torch.float64), b.to(torch.float64)
    wide_c, wide_s = cosine.to(torch.float64), sine.to(torch.float64)
    first = (a * cosine).to(torch.float64)     # round32(a*c)
    second = (b * sine).to(torch.float64)      # round32(b*s)
    cross_a = (a * sine).to(torch.float64)     # round32(a*s)
    cross_b = (b * cosine).to(torch.float64)   # round32(b*c)
    if component == 0:
        if spelling == "first":
            return (wide_a * wide_c - second).to(torch.float32)
        if spelling == "second":
            return (first - wide_b * wide_s).to(torch.float32)
        if spelling == "unfused":
            return (first - second).to(torch.float32)
    elif component == 1:
        if spelling == "first":
            return (wide_a * wide_s + cross_b).to(torch.float32)
        if spelling == "second":
            return (wide_b * wide_c + cross_a).to(torch.float32)
        if spelling == "unfused":
            return (cross_a + cross_b).to(torch.float32)
    else:  # pragma: no cover - internal misuse
        raise ValueError(f"component must be 0 (real) or 1 (imag), got {component}")
    raise ValueError(f"unknown spelling {spelling!r}")


def _candidate_bits(pairs: torch.Tensor, angles: torch.Tensor, spelling: str) -> torch.Tensor:
    """Both components under one spelling, ``[n, 2]`` FP32 -- kept for callers that
    only need the two well-known uniform spellings."""
    return torch.stack([_candidate_component(pairs, angles, spelling, 0),
                        _candidate_component(pairs, angles, spelling, 1)], dim=1)


def _probe_tensors(device="cpu"):
    target = torch.device(device)
    pairs = torch.tensor([(probe[0], probe[1]) for probe in _CONTRACTION_PROBES],
                         dtype=torch.float32, device=target)
    angles = torch.tensor([(probe[2], probe[3]) for probe in _CONTRACTION_PROBES],
                          dtype=torch.float32, device=target)
    return pairs, angles


def reference_contraction(device="cpu") -> dict[str, str]:
    """Which FMA spelling this host's ATen complex multiply implements, per component.

    Returns ``{"real": spelling, "imag": spelling}`` where ``spelling`` is one of
    :data:`SPELLINGS` or ``"unknown"``.

    This is a **measurement, not a switch**. It exists because the fused kernel has
    to commit to one arithmetic and that commitment belongs in the source, not in
    the machine: on the Ascend host's aarch64 build the real component measured
    ``"second"`` (``fma(-b, s, round32(a*c))``) while the imaginary component
    matched none of the three spellings, and an earlier x86 build measured
    ``"first"`` for both. A host whose measurement differs from
    :data:`DOCUMENTED_POLICY` does not share the fused kernel's arithmetic, so bit
    parity with that host's ATen is not a claim this operator can make there.

    Non-finite probe operands are not used: NaN payloads are not a contract.
    """
    pairs, angles = _probe_tensors(device)
    observed = _complex_bits(pairs, angles).to(torch.int32)
    result = {}
    for component, name in ((0, "real"), (1, "imag")):
        matches = [spelling for spelling in SPELLINGS
                   if torch.equal(_candidate_component(pairs, angles, spelling, component)
                                  .to(torch.int32), observed[:, component])]
        result[name] = matches[0] if len(matches) == 1 else "unknown"
    return result


def matches_documented_policy(device="cpu") -> bool:
    """Whether this host's reference implements the fused kernel's arithmetic.

    ``True`` means bit parity between the fused backend and this host's ATen
    complex multiply is a claim the tests can make; ``False`` means they differ by
    roughly an ulp and the fused backend keeps its own documented arithmetic
    without being compared to that host's ATen.
    """
    measured = reference_contraction(device)
    return (measured["real"], measured["imag"]) == DOCUMENTED_POLICY


def needs_64_bit_indices(batch: int, sequence: int, heads: int, width: int,
                         strides: tuple[int, int, int, int], tile_lanes: int) -> bool:
    """Whether the kernel's addresses can exceed ``int32`` for this geometry.

    Split out from the launcher so it is a plain integer function that can be
    exercised without a device. The answer must be *conservative*: a too-wide index
    costs a slower kernel, a too-narrow one silently addresses the wrong memory.

    The bound is the reachable offset including the masked tail lanes of the last
    width tile, because those lanes still form an address before the mask discards
    them -- and an address that needs 64 bits must never be computed in 32.
    """
    stride_batch, stride_sequence, stride_head, stride_width = strides
    reach = ((max(batch, 1) - 1) * stride_batch
             + (sequence - 1) * stride_sequence
             + (heads - 1) * stride_head
             + tile_lanes * stride_width)
    # The partner load is one width stride away from the lane's own offset.
    return reach + stride_width >= 2**31


def _overlaps(first: torch.Tensor, second: torch.Tensor) -> bool:
    structural = getattr(torch._C, "_overlaps", None)
    if structural is not None:
        return bool(structural(first, second))
    # Conservative fallback: refuse any shared storage rather than guess ranges.
    return first.untyped_storage().data_ptr() == second.untyped_storage().data_ptr()


def _injective(dims: tuple[int, ...], strides: tuple[int, ...]) -> bool:
    """Whether these dims/strides map distinct coordinates to distinct addresses.

    A view whose strides can carry -- two coordinates summing to the same offset --
    would let this in-place rewrite read an element another lane had already
    overwritten. That is invisible on a contiguous tensor and wrong on a
    mistaken transpose, so it is refused rather than assumed away.

    The test is sufficient, not merely necessary: taking the dims in increasing
    stride order, each stride must dominate the reach of everything smaller.
    Sliced views -- the case this operator exists for -- pass, because a slice
    only ever *increases* a stride. Broadcasting and transpose-like views fail.
    """
    bound = 1
    for stride, size in sorted(zip(strides, dims)):
        if size == 1:
            continue
        if stride < bound:
            return False
        bound = stride * size
    return True


def _checked(x, table, inverse):
    """Validate one call and return the geometry the two backends share.

    Everything a backend needs is decided here, once, so the Torch reference and
    the fused kernel cannot disagree about which tensor they were given.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if not isinstance(table, torch.Tensor):
        raise TypeError("table must be a torch.Tensor holding the (cos, sin) pairs")
    if not isinstance(inverse, bool):
        raise TypeError(f"inverse must be a bool, got {type(inverse).__name__}")
    if x.layout != torch.strided:
        raise ValueError("x must have a dense strided layout")
    if x.device.type not in {"cpu", "cuda", "npu"}:
        raise ValueError(f"x uses unsupported device {x.device}")
    if x.dtype not in {torch.bfloat16, torch.float32}:
        raise TypeError(f"x must be BF16 or FP32 activations, got {x.dtype}")
    if x.requires_grad:
        raise ValueError("the rotation is an inference operator and rewrites x in place")
    if x.ndim not in (3, 4):
        raise ValueError(
            f"x must be [batch, sequence, width] or [batch, sequence, heads, width], "
            f"got rank {x.ndim}"
        )
    if x.ndim == 4:
        batch, sequence, heads, width = (int(size) for size in x.shape)
        stride_batch, stride_sequence, stride_head, stride_width = x.stride()
    else:
        # Rank 3 is the model's [b, s, d] with a single implicit head -- the axis
        # order the runtime's own rotary uses (it broadcasts a length-one head for
        # the complex path). It is *not* [s, h, d]: the sequence axis is dim 1, so
        # a decode step is [b, 1, d] and its table is one row.
        batch, sequence, width = (int(size) for size in x.shape)
        heads = 1
        stride_batch, stride_sequence, stride_width = x.stride()
        # The head axis has size one, so its stride is never used; zero keeps the
        # kernel's single row expression valid.
        stride_head = 0
    if width <= 0 or width % 2:
        raise ValueError(f"the last axis is the rotated width and must be even, got {width}")
    if heads <= 0 or sequence <= 0 or batch < 0:
        raise ValueError(
            f"x must have positive sequence and head axes, got "
            f"{tuple(int(size) for size in x.shape)}"
        )
    if min(stride_sequence, stride_width) <= 0 or (heads > 1 and stride_head <= 0):
        raise ValueError(
            "x must have positive strides on the sequence, head and width axes; a "
            "broadcast or transposed view cannot be rotated in place"
        )
    if batch > 1 and stride_batch <= 0:
        # A broadcast batch axis would make several rows write one address, and the
        # row loop assumes distinct rows are distinct memory.
        raise ValueError(
            f"x must have a positive batch stride when the batch axis is broadcast, "
            f"got {stride_batch}"
        )
    dims = (batch, sequence, heads, width)
    strides = (stride_batch, stride_sequence, stride_head, stride_width)
    if not _injective(dims, strides):
        raise ValueError(
            f"x's strides {strides} can map two coordinates onto one address, so an "
            "in-place rotation could read a value another lane already rewrote"
        )
    if table.layout != torch.strided or table.dtype != torch.float32:
        raise TypeError(f"table must be a dense FP32 tensor, got {table.dtype}")
    if not table.is_contiguous():
        raise ValueError("table must be contiguous; its (cos, sin) pairs are adjacent")
    if table.device != x.device:
        raise ValueError(f"table is on {table.device} but x is on {x.device}")
    # A decode step rotates one position, and its table is naturally a single row
    # ``[width // 2, 2]`` rather than a length-one stack. Accepting that form here
    # is what keeps the caller from having to know this op's rank conventions; it
    # is a view, so nothing is copied and the kernel still sees [1, width // 2, 2].
    if (sequence == 1 and table.ndim == 2 and tuple(table.shape) == (width // 2, 2)):
        normalized_table = table.unsqueeze(0)
    elif tuple(table.shape) == (sequence, width // 2, 2):
        normalized_table = table
    else:
        raise ValueError(
            f"table must be [sequence, width // 2, 2] = {(sequence, width // 2, 2)}"
            f"{' (or a single [width // 2, 2] row when sequence == 1)' if sequence == 1 else ''}"
            f", got {tuple(table.shape)}"
        )
    if _overlaps(x, table):
        raise ValueError("table must not share storage with x; x is rewritten in place")
    return batch, sequence, heads, width, (
        stride_batch, stride_sequence, stride_head, stride_width
    ), normalized_table


def _torch_rotate(x, table, inverse, batch, sequence, heads, width):
    """The same rotation through ATen's complex multiply, on x's own device.

    This is the oracle rather than a second implementation: it builds the
    ``complex64`` view the reference multiplies and lets ATen perform the
    contracted product, then writes the narrowed result back through x's
    strides. It therefore defines the answer instead of approximating it, and
    it materialises FP32 copies of x -- acceptable for the oracle path, which is
    the CPU/small-tensor route.
    """
    flat = x.detach().to(torch.float32, copy=True).contiguous()
    pairs = torch.view_as_complex(flat.reshape(batch, sequence, heads, width // 2, 2))
    angle = torch.view_as_complex(table.reshape(sequence, width // 2, 2))
    if inverse:
        angle = angle.conj()
    product = pairs * angle.view(1, sequence, 1, width // 2)
    rotated = torch.view_as_real(product).reshape(x.shape)
    x.copy_(rotated.to(x.dtype))
    return x


def _triton_rotate(x, table, inverse, batch, sequence, heads, width, strides):
    """Run the fused Ascend kernel on the caller's current stream and device."""
    if x.numel() == 0:
        return x
    try:
        import triton
    except ImportError as error:  # pragma: no cover - depends on the install
        raise RuntimeError("the fused rotary backend needs Triton, which is not importable") from error
    if not hasattr(triton.language, "fma"):
        # Named explicitly: an installed Triton that renamed the fused multiply-add
        # must be adapted deliberately, never silently replaced by `a * b + c`,
        # which is the unfused arithmetic this operator exists to rule out.
        raise RuntimeError(
            "the installed Triton exposes no tl.fma, and the fused rotary kernel must "
            "not fall back to a contraction the compiler chooses"
        )
    from ._ascend_rotary_kernels import rotary_pairs_kernel

    stride_batch, stride_sequence, stride_head, stride_width = strides
    rows = batch * sequence * heads if batch else 0
    k_tile = min(K_TILE_MAX, max(2, 1 << (width - 1).bit_length()))
    k_steps = (width + k_tile - 1) // k_tile
    programs = min(rows, MAX_PROGRAMS)
    index64 = needs_64_bit_indices(batch, sequence, heads, width, strides,
                                   k_steps * k_tile)
    # Reuses the fused-unpack NPU launch context: the same device scope, the same
    # "Triton's active target really is Ascend" check, no stream or sync changes.
    module = importlib.import_module("relic_core.kernels.ascend_unpack")
    with module._npu_launch(x):
        rotary_pairs_kernel[(programs,)](
            x, table, rows, width, width // 2, programs, sequence, heads,
            stride_batch, stride_sequence, stride_head, stride_width,
            K_TILE=k_tile, K_STEPS=k_steps, INVERSE=inverse, INDEX64=index64,
        )
    return x


def _select_backend(backend, device):
    if not isinstance(backend, str):
        raise TypeError(f"backend must be a string, got {type(backend).__name__}")
    if backend not in {"auto", "torch", "triton"}:
        raise ValueError(f"backend must be 'auto', 'torch' or 'triton', got {backend!r}")
    if backend == "auto":
        backend = "triton" if device.type == "npu" else "torch"
    if backend == "torch":
        if device.type == "npu":
            # Refused by name: the NPU is the device this kernel exists for, and a
            # silent complex64 detour there would hide which arithmetic ran.
            raise ValueError(
                "the Torch complex reference is for CPU/CUDA; on NPU request the fused "
                "backend or omit backend"
            )
        return "torch"
    if device.type != "npu":
        raise ValueError("the fused rotary backend is for NPU tensors only")
    if not supports_triton_ascend():
        raise RuntimeError(
            "the fused rotary backend needs an installed Triton Ascend backend"
        )
    return "triton"


@torch.no_grad()
def apply_rotary_inplace(x: torch.Tensor, table: torch.Tensor, *,
                         inverse: bool = False, backend: str = "auto") -> torch.Tensor:
    """Rotate the ``(real, imag)`` pairs of ``x`` in place and return ``x``.

    ``x`` is ``[batch, sequence, width]`` or ``[batch, sequence, heads, width]``
    of BF16 or FP32, with an even positive last axis. The rank-3 form is the
    model's own ``[b, s, d]`` with a single implicit head, so the sequence axis is
    dim 1 in both ranks -- a decode step is ``[b, 1, d]``. ``x`` may be a strided
    view -- a head slice of a fused QKV buffer is the case this exists for --
    provided its strides cannot map two coordinates onto one address. ``table`` is
    contiguous FP32 ``[sequence, width // 2, 2]``, ``(cos, sin)`` adjacent, on the
    same device, and may be a single ``[width // 2, 2]`` row when the sequence axis
    has length one -- the decode case, where a step rotates one position. It must
    not share storage with ``x``.

    ``inverse=True`` rotates by the conjugate (the same multiply with a negated
    sine), which is the reverse direction. ``backend`` is ``"torch"`` for ATen's
    complex multiply (the CPU/CUDA oracle), ``"triton"`` for the fused Ascend
    kernel, or ``"auto"``: fused on NPU, ATen elsewhere. No backend falls back to
    the other after being chosen, and no contraction is left to the compiler.

    The fused backend implements :data:`DOCUMENTED_POLICY` -- both products
    rounded to FP32, the first product of each component fused -- which is an
    explicit arithmetic rather than a claim about any particular host's ATen. The
    ``"torch"`` backend *is* ATen's complex multiply, so it matches the local
    reference by construction; the fused backend is held to the exact-rational
    ``fmaf`` model of the documented policy, and to a host's ATen only when
    :func:`matches_documented_policy` reports that host implements the same
    arithmetic. On a host that contracts differently the two differ by about an
    ulp and no parity is claimed.
    """
    batch, sequence, heads, width, strides, angle_table = _checked(x, table, inverse)
    implementation = _select_backend(backend, x.device)
    if implementation == "torch":
        return _torch_rotate(x, angle_table, inverse, batch, sequence, heads, width)
    return _triton_rotate(x, angle_table, inverse, batch, sequence, heads, width, strides)