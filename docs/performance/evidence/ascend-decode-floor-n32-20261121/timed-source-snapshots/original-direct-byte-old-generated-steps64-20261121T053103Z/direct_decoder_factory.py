"""Explicit private choice of original-byte consumer; no runtime default replacement."""
from __future__ import annotations

import os

from relic_core.kernels.ascendc_online_tile_cube import AscendCTileCubeConsumer
from relic_core.kernels.direct_aiv_gemv import DirectAIVConsumer


def make_consumer(library_path, *, sha256, block_dim=24):
    mode = os.environ["RELIC_DIRECT_CONSUMER_MODE"]
    if mode == "aiv":
        return DirectAIVConsumer(block_dim=block_dim)
    if mode not in ("bits", "old", "prepared", "prepared-bits"):
        raise ValueError("select only the independently probed consumer")
    operator = AscendCTileCubeConsumer(library_path, sha256=sha256, block_dim=block_dim)
    if (operator.direct_bf16_bits is not (mode in ("bits", "prepared-bits"))
            or operator.prepared_decode_tables is not mode.startswith("prepared")):
        raise RuntimeError("the selected native decoder does not match its recorded policy")
    return operator
