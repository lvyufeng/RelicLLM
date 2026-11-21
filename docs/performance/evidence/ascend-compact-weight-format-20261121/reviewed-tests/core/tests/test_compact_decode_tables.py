"""Contracts for the compact Cube-order decode tables and the format/count queries.

These pin the parts of the compact handoff that a wrong value would not crash on: the
table extent, the scale-index fill, and the ABI queries that decide which format the
compiled kernel expects. The numeric gates already prove that a correct bank produces
correct bits; what is untested there is the *plumbing* around it.

Three independent statements are used, none of which reads a value out of the code under
test to build its expectation:

* the **table extent** is recomputed from the native side's own section arithmetic
  (``online_tile_cube.cpp`` lines 53-59) rather than from the Python helper;
* the **scale index** is derived from the documented compact byte order
  ``[N/32, K/128, 8, 2, 16, 8]`` -- pick a value index, recover its
  ``(k16, n16, row, kpair)``, and compute ``group * 32 + nrow`` -- and is *also* compared
  against the native scalar fill formula ``(i / (N * K_BLOCK)) * N + (i / 16) % N``, so the
  two statements have to agree with each other and with the table;
* the **lookup words** are the binary32 encodings of the eight E2M1 magnitudes written out
  from the encoding rule, not copied from the module's unit table.

One deliberate non-contract, recorded so it is not mistaken for covered: *nothing here
certifies that a weight bank is in the format the library was compiled for*. The legacy and
compact banks are byte-for-byte the same total size (pinned below), so no shape check can
tell them apart; that gap is open pending an explicit bank-format argument, and
``test_legacy_and_compact_banks_have_the_same_total_bytes`` exists only to keep the reason
visible.

No device, no build, no kernel launch: the fake library below is a namespace of query
callables and is never entered.
"""
from __future__ import annotations

from types import SimpleNamespace
import hashlib

import pytest

torch = pytest.importorskip("torch")

from relic_core.kernels import ascendc_online_tile_cube as native  # noqa: E402

#: Compact geometry the C++ static_assert requires: N32 output rows, four K32 blocks per
#: tile (K128), floating decoder.
CUBE_GROUP_K = 4
CUBE_ROWS = 32
#: 128 K values hold 64 packed code bytes, so a 128-byte scale block per (N32, K128) tile.
CUBE_SCALE_BYTES = CUBE_ROWS * CUBE_GROUP_K
#: Table sections, restated from `online_tile_cube.cpp`: 256 lookup + 256 scale lookup,
#: then one repeat entry per decoded value (or per packed byte under a direct-bit decoder),
#: then, for the legacy order only, one NZ index per packed byte.
LOOKUP_WORDS = 256
SCALE_LOOKUP_WORDS = 256
HEADER_WORDS = LOOKUP_WORDS + SCALE_LOOKUP_WORDS


def _native_table_words(*, cube_packed, group_k, output_rows, direct_bits):
    """``TABLE_WORDS`` as the kernel computes it, written out independently of Python."""
    k_block = 32
    block_k = k_block * group_k
    elements = output_rows * block_k
    packed_bytes = elements // 2
    repeat = HEADER_WORDS
    repeat_count = packed_bytes if direct_bits else elements
    table_nz = repeat + repeat_count
    table_mask = table_nz + packed_bytes
    chosen = table_nz if cube_packed else table_mask
    return chosen + (512 if direct_bits else 0)


def _compact_scale_index(value_index: int) -> int:
    """Scale element for a decoded value, derived from the compact byte order.

    The code block is ``[nb, kb, k16, n16, row, kpair]`` with one packed byte per
    ``kpair``, two codes per byte, low nibble first. So for value ``i`` the packed byte is
    ``b = i // 2`` and

    * ``kpair = b % 8``, ``row = (b // 8) % 16``, ``n16 = (b // 128) % 2``, ``k16 = b // 256``;
    * the K position is ``k16 * 16 + kpair * 2 + nibble``, so its 32-wide scale group is
      ``k16 // 2``;
    * the N row is ``n16 * 16 + row``;
    * the scale tensor is ``[nb, kb, group, nrow]``, hence ``group * 32 + nrow``.
    """
    packed_byte = value_index // 2
    kpair = packed_byte % 8
    row = (packed_byte // 8) % 16
    n16 = (packed_byte // 128) % 2
    k16 = packed_byte // 256
    assert kpair < 8 and row < 16 and n16 < 2 and k16 < 8
    group = k16 // 2
    nrow = n16 * 16 + row
    return group * CUBE_ROWS + nrow


def _native_scale_index_formula(value_index: int) -> int:
    """The scalar fill used by the kernel when no prepared table is supplied."""
    return (value_index // (CUBE_ROWS * 32)) * CUBE_ROWS + (value_index // 16) % CUBE_ROWS


def _e2m1_bf16_units() -> list[int]:
    """The sixteen E2M1 codes as BF16 bit patterns, from the magnitude rule.

    Magnitudes are ``0, 0.5, 1, 1.5, 2, 3, 4, 6``; their BF16 encodings are the same
    patterns at exponent 127, and codes 8..15 are the negations. Code 8 (the negative zero)
    folds to +0, which is what the accepted I8x2 widening does.
    """
    magnitudes = [0x0000, 0x3F00, 0x3F80, 0x3FC0, 0x4000, 0x4040, 0x4080, 0x40C0]
    units = list(magnitudes)
    units += [0x0000 if value == 0 else (value | 0x8000) for value in magnitudes]
    return units


def _table_words(cube_packed: bool, group_k: int = CUBE_GROUP_K, rows: int = CUBE_ROWS):
    table = native._decode_table_cpu(group_k, False, rows, cube_packed)
    return (table.long() & 0xFFFFFFFF).tolist()


# ------------------------------------------------------------------------------ extents


def test_compact_table_extent_is_4608_and_the_n32_legacy_extent_is_6656():
    """The two extents the ABI handoff hinges on, from both statements.

    A compact build allocates no NZ index section, so its table is exactly one
    ``packed_bytes`` tail shorter than the N32 legacy table for the same geometry.
    """
    compact = native._decode_table_words(CUBE_GROUP_K, False, CUBE_ROWS, True)
    legacy_n32 = native._decode_table_words(CUBE_GROUP_K, False, CUBE_ROWS, False)
    assert compact == CUBE_GROUP_K * CUBE_ROWS * 32 + HEADER_WORDS == 4096 + 512 == 4608
    assert legacy_n32 == compact + CUBE_ROWS * CUBE_GROUP_K * 32 // 2 == 4608 + 2048 == 6656
    assert compact == _native_table_words(
        cube_packed=True, group_k=CUBE_GROUP_K, output_rows=CUBE_ROWS, direct_bits=False
    )
    assert legacy_n32 == _native_table_words(
        cube_packed=False, group_k=CUBE_GROUP_K, output_rows=CUBE_ROWS, direct_bits=False
    )


@pytest.mark.parametrize(
    ("group_k", "rows", "direct_bits"),
    # `output_rows * group_k <= 128` bounds the grid, so e.g. 8 blocks x 32 rows is not
    # admissible and is refused by the first guard rather than tabulated.
    [(4, 32, False), (4, 16, False), (2, 16, False), (1, 16, False),
     (8, 16, False), (4, 32, True), (4, 16, True)],
)
def test_every_admissible_geometry_agrees_with_the_native_section_arithmetic(
    group_k, rows, direct_bits
):
    """Not just the compact point: the whole admissible grid of the helper."""
    for cube_packed in (False, True):
        if cube_packed and (group_k, rows, direct_bits) != (CUBE_GROUP_K, CUBE_ROWS, False):
            continue
        assert native._decode_table_words(group_k, direct_bits, rows, cube_packed) == (
            _native_table_words(
                cube_packed=cube_packed,
                group_k=group_k,
                output_rows=rows,
                direct_bits=direct_bits,
            )
        ), (group_k, rows, direct_bits, cube_packed)


@pytest.mark.parametrize(
    ("group_k", "rows", "cube_packed"),
    [(4, 32, True), (4, 32, False), (4, 16, False)],
)
def test_the_built_table_length_is_exactly_the_declared_extent(group_k, rows, cube_packed):
    """A short table would make the kernel gather past the buffer; pin the length."""
    declared = native._decode_table_words(group_k, False, rows, cube_packed)
    assert _table_words(cube_packed, group_k, rows).__len__() == declared


def test_the_compact_table_stops_before_the_nz_tail():
    """In compact order there is no NZ gather left to index, so the tail is absent.

    Only the *extent* shrinks by the NZ section; the repeat fill is a different mapping on
    purpose, because the compact order addresses values by tile coordinates while the
    legacy order walks K blocks linearly. Asserting the two repeat sections equal would be
    asserting the two layouts coincide, so each is checked against its own statement
    instead.
    """
    compact = _table_words(True)
    legacy = _table_words(False)
    elements = CUBE_ROWS * 32 * CUBE_GROUP_K
    assert len(compact) == HEADER_WORDS + elements
    assert len(legacy) == HEADER_WORDS + elements + elements // 2
    # The header sections are pure byte->word metadata and do not depend on the value order.
    assert compact[:HEADER_WORDS] == legacy[:HEADER_WORDS]
    # The repeat sections are deliberately different mappings.
    assert compact[HEADER_WORDS:] != legacy[HEADER_WORDS:HEADER_WORDS + elements]
    # ... and each is correct in its own right.
    for value_index, word in enumerate(compact[HEADER_WORDS:]):
        assert word == _compact_scale_index(value_index) * 4, value_index
    for value_index, word in enumerate(legacy[HEADER_WORDS:HEADER_WORDS + elements]):
        assert word == (value_index // 32) * 4, value_index


# -------------------------------------------------------------------------- scale index


def test_compact_scale_index_follows_the_documented_coordinates():
    """Every repeat entry against the layout-derived scale index, for all 4096 values."""
    table = _table_words(True)
    repeat = table[HEADER_WORDS:]
    assert len(repeat) == CUBE_ROWS * 32 * CUBE_GROUP_K
    for value_index, word in enumerate(repeat):
        assert word == _compact_scale_index(value_index) * 4, value_index


def test_compact_scale_index_agrees_with_the_native_scalar_fill():
    """The kernel's own fallback formula and the coordinate derivation must coincide.

    This is the statement that ties the prepared table to the path taken when no table is
    supplied; if either side ever changes, they disagree here rather than on the card.
    """
    for value_index in range(CUBE_ROWS * 32 * CUBE_GROUP_K):
        assert _compact_scale_index(value_index) == _native_scale_index_formula(value_index)


def test_compact_scale_indexes_stay_inside_the_128_byte_scale_block():
    """One N32/K128 tile has 128 scale bytes, and every index must land inside them."""
    repeat = _table_words(True)[HEADER_WORDS:]
    indexes = [word // 4 for word in repeat]
    assert min(indexes) == 0
    assert max(indexes) == CUBE_SCALE_BYTES - 1
    # Every row and every group is reachable, and each of the 128 slots is used exactly 32
    # times: 4096 values / 128 scale slots, since a scale covers 32 K values.
    counts = [0] * CUBE_SCALE_BYTES
    for index in indexes:
        counts[index] += 1
    assert set(counts) == {CUBE_ROWS}


def test_compact_scale_index_groups_the_low_and_high_nibble_of_one_byte_together():
    """Both codes of a packed byte are 16 apart in K, so they share one K32 scale group."""
    repeat = _table_words(True)[HEADER_WORDS:]
    for packed_byte in range(CUBE_ROWS * 32 * CUBE_GROUP_K // 2):
        low = repeat[2 * packed_byte] // 4
        high = repeat[2 * packed_byte + 1] // 4
        assert low == high, packed_byte


# ------------------------------------------------------------------------------ lookups


def test_compact_lookup_covers_all_256_bytes_and_folds_the_negative_zero():
    """Header word i decodes byte ``(i + 128) & 255``; check all 256 against the rule."""
    table = _table_words(True)
    units = _e2m1_bf16_units()
    assert units[8] == 0x0000  # code 8 is -0.0 and folds to +0
    for index in range(LOOKUP_WORDS):
        byte = (index + 128) & 255
        expected = units[byte & 15] | (units[byte >> 4] << 16)
        assert table[index] == expected, byte


def test_compact_scale_lookup_holds_the_e8m0_band_and_poisons_the_rest():
    """The accepted scale band is 2..252; a byte outside it must poison, not saturate."""
    table = _table_words(True)
    for index in range(SCALE_LOOKUP_WORDS):
        byte = (index + 128) & 255
        expected = (byte << 23) if 2 <= byte <= 252 else 0x7FC00000
        assert table[SCALE_LOOKUP_WORDS + index] == expected, byte
    band = [byte for byte in range(256) if 2 <= byte <= 252]
    assert len(band) == 251 and 0 not in band and 1 not in band and 252 in band


# ------------------------------------------------------------------- format / ABI gates


@pytest.mark.parametrize(
    ("group_k", "direct_bits", "rows"),
    [
        (4, False, 16),  # compact order needs the N32 tile
        (2, False, 16),  # ... and the K128 grouping
        (1, False, 32),
        (8, False, 32),
        (4, True, 32),  # compact order is only defined for the floating decoder
    ],
)
def test_compact_tables_are_refused_outside_the_floating_n32_k128_abi(
    group_k, direct_bits, rows
):
    """Refuse every non-Cube geometry before producing any metadata table."""
    # Builder and native-library admission have distinct messages; here the
    # contract is that neither table builder accepts an unsupported geometry.
    with pytest.raises(ValueError):
        native._decode_table_words(group_k, direct_bits, rows, True)
    with pytest.raises(ValueError):
        native._decode_table_cpu(group_k, direct_bits, rows, True)


@pytest.mark.parametrize("cube_packed", [1, 0, None, "yes"])
def test_the_compact_flag_itself_must_be_a_bool(cube_packed):
    """A truthy int would silently select the wrong table length."""
    with pytest.raises(ValueError):
        native._decode_table_words(CUBE_GROUP_K, False, CUBE_ROWS, cube_packed)


def test_a_compact_artifact_without_the_table_words_query_is_refused(
    tmp_path, monkeypatch
):
    """A prepared-table artifact has to publish its extent; the reader will not guess it.

    Without the query the reader cannot tell 4608 from 6656, which is exactly the pair that
    a compact/legacy mix would get wrong, so the incomplete ABI is refused.
    """
    path = tmp_path / "compact-without-extent.so"
    path.write_bytes(b"verified fake compact ABI missing the extent query; never executed")
    symbols = {
        "relic_online_tile_cube_launch": _NeverLaunched(),
        "relic_online_tile_cube_k_blocks_per_tile": _Query(CUBE_GROUP_K),
        "relic_online_tile_cube_output_rows": _Query(CUBE_ROWS),
        "relic_online_tile_cube_direct_bf16_bits": _Query(0),
        "relic_online_tile_cube_compact_layout": _Query(1),
        "relic_online_tile_cube_prepared_tables": _Query(1),
        "relic_online_tile_cube_launch_tables": _NeverLaunched(),
    }
    monkeypatch.setattr(native.ctypes, "CDLL", lambda _: SimpleNamespace(**symbols))
    with pytest.raises(ValueError, match="complete checked table-launch ABI"):
        native.AscendCTileCubeConsumer(
            path, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )


def test_legacy_and_compact_banks_have_the_same_total_bytes():
    """Why a bank/format mix cannot be caught by a shape check.

    A legacy bank and a compact bank for the same expert are the same size -- the compact
    layout trades N16 tiles for the N32 halving and K32 blocks for the K128 grouping, which
    cancels exactly. So the consumer sees an identical ``[owned, tiles, blocks, 16, 16]``
    view either way, and the format has to be certified out of band.
    """
    for tiles, blocks in ((2, 4), (4, 4), (8, 8), (16, 32), (32, 8)):
        blocks_per_tile = blocks // CUBE_GROUP_K
        legacy = tiles * blocks_per_tile * (16 * (32 * CUBE_GROUP_K) // 2)
        compact = (tiles // 2) * blocks_per_tile * (32 * (32 * CUBE_GROUP_K) // 2)
        assert legacy == compact, (tiles, blocks)
        # The host-side views that `_validate` checks are identical for both formats.
        assert tiles * blocks * (16 * 16) == legacy
        assert tiles * blocks * 16 == (tiles // 2) * blocks_per_tile * CUBE_SCALE_BYTES


# ------------------------------------------------------------------------ library queries


class _Query:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value


class _NeverLaunched:
    def __call__(self, *args):
        pytest.fail("format-query tests must not launch a native kernel")


def _operator(tmp_path, monkeypatch, *, compact=None, rows=16, group_k=4, bits=0,
              prepared=None, table_words=None, unsigned=None):
    """A verified-checksum stand-in library whose symbols are query callables only."""
    path = tmp_path / "compact-query-only.so"
    path.write_bytes(b"verified fake compact query ABI; never executed")
    symbols = {
        "relic_online_tile_cube_launch": _NeverLaunched(),
        "relic_online_tile_cube_k_blocks_per_tile": _Query(group_k),
        "relic_online_tile_cube_output_rows": _Query(rows),
        "relic_online_tile_cube_direct_bf16_bits": _Query(bits),
    }
    if compact is not None:
        symbols["relic_online_tile_cube_compact_layout"] = _Query(compact)
    if unsigned is not None:
        symbols["relic_online_tile_cube_unsigned_byte_cast"] = _Query(unsigned)
    if prepared is not None:
        symbols["relic_online_tile_cube_prepared_tables"] = _Query(prepared)
        symbols["relic_online_tile_cube_table_words"] = _Query(table_words)
        symbols["relic_online_tile_cube_launch_tables"] = _NeverLaunched()
    monkeypatch.setattr(native.ctypes, "CDLL", lambda _: SimpleNamespace(**symbols))
    return native.AscendCTileCubeConsumer(
        path, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )


def test_a_library_without_the_compact_query_stays_legacy(tmp_path, monkeypatch):
    """Older artifacts must keep loading, and must not be relabelled as compact."""
    op = _operator(tmp_path, monkeypatch, compact=None)
    assert op.cube_packed is False
    assert op.weight_format == "original-n16"
    assert op.describe()["weight_format"] == "original-n16"
    assert op.describe()["resident_encoding_changed"] is False


def test_a_compact_library_reports_the_recorded_format(tmp_path, monkeypatch):
    op = _operator(tmp_path, monkeypatch, compact=1, rows=32)
    assert op.cube_packed is True
    assert op.weight_format == "relic.fp4.cube-n32-k128.v1"
    assert op.describe()["weight_format"] == "relic.fp4.cube-n32-k128.v1"
    assert op.describe()["resident_encoding_changed"] is True
    assert op.describe()["source_checkpoint_files_unchanged"] is True


def test_a_legacy_library_is_not_relabelled_compact(tmp_path, monkeypatch):
    op = _operator(tmp_path, monkeypatch, compact=0)
    assert op.cube_packed is False
    assert op.weight_format == "original-n16"


@pytest.mark.parametrize("compact", [2, 3, 0xFFFFFFFF])
def test_a_non_boolean_compact_query_is_refused(tmp_path, monkeypatch, compact):
    """The mode selects a byte order; an unexpected value must not pick one."""
    with pytest.raises(ValueError, match="explicit boolean"):
        _operator(tmp_path, monkeypatch, compact=compact)


@pytest.mark.parametrize(
    ("rows", "group_k", "bits"),
    [(16, 4, 0), (32, 2, 0), (32, 4, 1), (32, 8, 0)],
)
def test_a_compact_library_must_also_report_the_floating_n32_k128_abi(
    tmp_path, monkeypatch, rows, group_k, bits
):
    """A compact flag under any other decoder is an inconsistent artifact."""
    with pytest.raises(ValueError, match="checked N32/K128 floating ABI|output-row geometry"):
        _operator(tmp_path, monkeypatch, compact=1, rows=rows, group_k=group_k, bits=bits)


def test_a_compact_library_declaring_the_legacy_table_extent_is_refused(
    tmp_path, monkeypatch
):
    """The table extent is the cheapest place for a stale/legacy artifact to show up."""
    op = _operator(tmp_path, monkeypatch, compact=1, rows=32, prepared=1,
                   table_words=4608)
    assert op.prepared_decode_tables is True
    assert op.table_words == 4608
    with pytest.raises(ValueError, match="table extent differs"):
        _operator(tmp_path, monkeypatch, compact=1, rows=32, prepared=1, table_words=6656)


def test_a_legacy_library_declaring_the_compact_table_extent_is_refused(
    tmp_path, monkeypatch
):
    """And the same in the other direction: a compact extent under the legacy order."""
    op = _operator(tmp_path, monkeypatch, compact=0, rows=32, prepared=1, table_words=6656)
    assert op.cube_packed is False
    assert op.table_words == 6656
    with pytest.raises(ValueError, match="table extent differs"):
        _operator(tmp_path, monkeypatch, compact=0, rows=32, prepared=1, table_words=4608)

# ---------------------------------------------------------------- unsigned byte cast
#
# The `unsigned_byte_cast` policy removes the `+128` rotation from the table fill and the
# two `Adds(..., 128)` from the kernel: the A2 cast is exact for the whole uint8 range, so a
# byte indexes its own entry. The two policies must therefore describe the *same* byte
# values, only addressed differently -- otherwise the "no numeric change" claim is false.


def test_unsigned_table_indexes_bytes_zero_to_255_directly():
    """Header word i decodes byte i itself, not ``(i + 128) & 255``."""
    table = _table_words_unsigned()
    units = _e2m1_bf16_units()
    for byte in range(256):
        assert table[byte] == units[byte & 15] | (units[byte >> 4] << 16), byte
        expected_scale = (byte << 23) if 2 <= byte <= 252 else 0x7FC00000
        assert table[SCALE_LOOKUP_WORDS + byte] == expected_scale, byte


def test_unsigned_and_signed_tables_hold_the_same_value_for_the_same_byte():
    """The rotation is only addressing: NUX[b] must equal legacy[(b + 128) & 255].

    This is the statement behind `unsigned_byte_cast=False` staying numeric-identical, and
    the reason the flag cannot change a decoded value.
    """
    unsigned = _table_words_unsigned()
    signed = _table_words(True)
    for byte in range(256):
        assert unsigned[byte] == signed[(byte + 128) & 255], byte
        assert unsigned[SCALE_LOOKUP_WORDS + byte] == signed[
            SCALE_LOOKUP_WORDS + ((byte + 128) & 255)
        ], byte


def test_the_default_unsigned_flag_preserves_the_legacy_byte_order():
    """`unsigned_byte_cast` defaults to False, so an old caller sees the old tables."""
    # The table is int32 storage: mask each word to unsigned before comparing it against a
    # 32-bit pattern, otherwise any word with the top bit set (e.g. 0xc0c0c0c0 for byte
    # 0xFF, 0xbfc00000 for byte 0xB8) reads back negative and never matches.
    default = _table_words(True)
    explicit_false = (
        native._decode_table_cpu(
            CUBE_GROUP_K, False, CUBE_ROWS, True, unsigned_byte_cast=False
        ).long()
        & 0xFFFFFFFF
    ).tolist()
    assert default == explicit_false
    units = _e2m1_bf16_units()

    def word_for(byte: int) -> int:
        return units[byte & 15] | (units[byte >> 4] << 16)

    # Legacy addressing: word i holds byte (i + 128) & 255. Both nibbles non-zero, so a
    # rotation that happens to land on a zero word cannot pass.
    for byte in (0x38, 0x4C, 0x71, 0xFF):
        assert default[(byte + 128) & 255] == word_for(byte), byte
    # ... and the same table read as unsigned would put the byte at its own index.
    assert default[0x38] == word_for((0x38 + 128) & 255) != word_for(0x38)


def test_unsigned_does_not_change_the_table_extent():
    """The extent check cannot detect a rotation mismatch, so the flag must not move it."""
    for group_k, rows, cube_packed in ((4, 32, True), (4, 32, False), (4, 16, False)):
        signed = native._decode_table_cpu(group_k, False, rows, cube_packed).numel()
        unsigned = native._decode_table_cpu(
            group_k, False, rows, cube_packed, unsigned_byte_cast=True
        ).numel()
        assert signed == unsigned == native._decode_table_words(
            group_k, False, rows, cube_packed
        )


def test_unsigned_requires_the_floating_decoder_in_the_builder():
    """Unsigned byte indices are only defined where the byte is not a signed cast."""
    with pytest.raises(ValueError, match="unsigned byte tables require"):
        native._decode_table_cpu(CUBE_GROUP_K, True, CUBE_ROWS, False, unsigned_byte_cast=True)
    with pytest.raises(ValueError):
        native._decode_table_cpu(
            CUBE_GROUP_K, False, CUBE_ROWS, False, unsigned_byte_cast=1
        )


def test_unsigned_is_allowed_together_with_the_compact_order():
    """Both flags are independent; the compact table simply indexes bytes directly."""
    table = native._decode_table_cpu(
        CUBE_GROUP_K, False, CUBE_ROWS, True, unsigned_byte_cast=True
    )
    assert table.numel() == 4608
    unsigned = (table.long() & 0xFFFFFFFF).tolist()
    units = _e2m1_bf16_units()
    for byte in range(256):
        assert unsigned[byte] == units[byte & 15] | (units[byte >> 4] << 16)


def _table_words_unsigned(group_k: int = CUBE_GROUP_K, rows: int = CUBE_ROWS):
    table = native._decode_table_cpu(group_k, False, rows, True, unsigned_byte_cast=True)
    return (table.long() & 0xFFFFFFFF).tolist()


def test_the_unsigned_query_is_optional_and_defaults_to_legacy(tmp_path, monkeypatch):
    """An artifact built without the new flag must keep the old byte order."""
    op = _operator(tmp_path, monkeypatch, compact=0)
    assert op.unsigned_byte_cast is False
    assert op.describe()["exact_A2_unsigned_byte_to_half_indices"] is False


def test_the_unsigned_query_is_reported_when_enabled(tmp_path, monkeypatch):
    op = _operator(tmp_path, monkeypatch, compact=0, unsigned=1, prepared=1,
                   table_words=3584)
    assert op.unsigned_byte_cast is True
    assert op.describe()["exact_A2_unsigned_byte_to_half_indices"] is True


@pytest.mark.parametrize("unsigned", [2, 3, 0xFFFFFFFF])
def test_a_non_boolean_unsigned_query_is_refused(tmp_path, monkeypatch, unsigned):
    with pytest.raises(ValueError, match="explicit floating policy"):
        _operator(tmp_path, monkeypatch, compact=0, unsigned=unsigned, prepared=1,
                  table_words=3584)


def test_unsigned_without_prepared_tables_is_refused(tmp_path, monkeypatch):
    """The policy is a table-fill choice, so it cannot stand without the tables."""
    with pytest.raises(ValueError, match="immutable prepared tables"):
        _operator(tmp_path, monkeypatch, compact=0, unsigned=1, prepared=0)


def test_unsigned_with_the_direct_bit_decoder_is_refused(tmp_path, monkeypatch):
    """Direct-bit decoding bypasses the byte-index tables, so the pair is incoherent."""
    with pytest.raises(ValueError, match="explicit floating policy"):
        _operator(tmp_path, monkeypatch, compact=0, unsigned=1, prepared=1,
                  table_words=3072, bits=1)


def test_unsigned_compact_tables_are_not_confused_with_legacy_extents(
    tmp_path, monkeypatch
):
    """A compact+unsigned artifact still declares 4608, not 6656."""
    op = _operator(tmp_path, monkeypatch, compact=1, rows=32, unsigned=1, prepared=1,
                   table_words=4608)
    assert op.cube_packed is True and op.unsigned_byte_cast is True
    assert op.weight_format == "relic.fp4.cube-n32-k128.v1"
    with pytest.raises(ValueError, match="table extent differs"):
        _operator(tmp_path, monkeypatch, compact=1, rows=32, unsigned=1, prepared=1,
                  table_words=6656)


# ------------------------------------------------------------------- the UB slot budget


def test_the_second_compact_bf16_slot_fits_the_unused_nz_index_space():
    """Two BF16 slots occupy exactly the bytes the legacy single slot plus NZ index used.

    The offsets are recomputed here from the kernel's own layout constants, so a future
    move that grows the Vector scratch has to prove it again rather than assume it.
    """
    n, group_k = 32, 4
    block_k = 32
    k = block_k * group_k
    packed_bytes = n * k // 2
    elements = n * k
    scale_count = n * group_k
    scale_pad = (scale_count + 31) // 32 * 32

    decoded_offset = packed_bytes + scale_pad
    half_offset = decoded_offset + 2 * elements * 2
    index_offset = half_offset + packed_bytes * 2
    lookup_offset = index_offset + packed_bytes * 4
    pairs_offset = lookup_offset + 256 * 4
    values_offset = pairs_offset + packed_bytes * 4
    scaled_offset = values_offset + elements * 4
    nz_offset = scaled_offset + elements * 2
    scale_half_offset = nz_offset + packed_bytes * 4

    two_slots = scaled_offset + 2 * elements * 2
    assert two_slots == scale_half_offset  # exactly abuts the next live tensor
    assert two_slots == nz_offset + packed_bytes * 4  # slot 1 is the whole NZ index space
    nz_index_bytes = packed_bytes * 4  # uint32 index words
    assert two_slots - scaled_offset == elements * 2 + nz_index_bytes
    # The legacy path still allocates one slot, so no mode pays for the other's layout.
    assert nz_offset - scaled_offset == elements * 2
