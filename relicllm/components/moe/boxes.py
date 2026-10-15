"""The two boxes this package prices a fit against, in one place.

These numbers were spelled out as literals in a dozen signatures, and the literals did not all
describe the same machine. That is the point of this module: there are two boxes, they are
different, and a default should say which one it is.

* :data:`POSITIONING_GPU_COUNT`, :data:`POSITIONING_GPU_MEMORY_GIB` are the box a model has to
  fit. The project's positioning is the older end of the accelerator market, and the two targets
  it starts with are both 32 GiB cards -- an Ascend 910A and a Tesla V100S. The card count is
  eight because that is the machine 910A work actually moved to: the account's own eight-card box.
  The *rental* catalog tops out at a four-card 910A spec, and capping a fit there would price a
  model at half the cards the model would run on.
* :data:`DEV_BOX_GPU_COUNT`, :data:`DEV_BOX_MEMORY_GIB` are the box the recorded numbers were
  taken on: four cards that report 22,528 MiB, which is the 22.0 GiB every threshold and every
  measured Xing4 margin in this tree was measured against. It stays as a named box because a
  question about *this* checkout's cards is a different question from a question about the
  targeted hardware, and a planner that answers the second is wrong about the first.

So a default that prices the hardware the project targets is the positioning box, and a default
that prices the machine the code is running on is the dev box. `placement.HardwareProfile` and
`triage.hardware_profile_for` are the first kind; `moe_planning`'s two plan builders and
`gguf.generation`'s config are the second, and they keep the dev box's numbers.

**What the positioning box does and does not decide.** It is the card side of the gate, and the
gate is not a parameter count. A model is in scope when its per-rank *resident* bytes fit one of
those cards **and** the off-card part of its checkpoint -- routed experts, gather tables -- fits
host memory that the runtime can actually reach, since that part is staged from the host rather
than held. DeepSeek-V4.1-Flash is what that means in practice: 475.24 GiB of checkpoint, 268.95
GiB of it routed experts and 189.13 GiB of it two Engram tables, 16.79 GiB retained on the cards.
The coarse rule of thumb the positioning states is MoE to about 500B parameters and dense to about
200B, and it is a statement about which *new* models are worth adapting rather than about what is
supported: the flagship this project adapts is 552B, above the MoE line.
"""

from __future__ import annotations

#: The box a model has to fit: the account's own eight-card 910A machine, and the 32 GiB cards the
#: project's two starting targets both are.
POSITIONING_GPU_COUNT = 8
POSITIONING_GPU_MEMORY_GIB = 32.0
POSITIONING_BOX_NAME = "8x32gib-positioning"

#: The box the recorded runs were taken on: four cards that report 22,528 MiB, so 22.0 GiB a card.
#: That is the size every threshold in this tree was measured against -- Xing4's 17.94 GiB resident
#: against 18.70 GiB of a 22 GiB card's usable share, and the baselines under ``tests/fixtures/perf``.
#: It is a named box rather than a second constant in disguise because a question about these cards
#: and a question about the targeting hardware are different questions, and a caller has to say
#: which one it is asking.
DEV_BOX_GPU_COUNT = 4
DEV_BOX_MEMORY_GIB = 22.0
DEV_BOX_NAME = "consumer-gpu-box"

