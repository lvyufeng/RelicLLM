# The positioning, and the box a fit is priced against

RelicLLM's positioning is the older end of the accelerator market: a model is worth running here when
a modern part would have to split it across cards, and the two targets it starts with are both
32 GiB cards -- an **Ascend 910A** and a **Tesla V100**, rented by the card-hour
([the fleet](openi_accelerator_fleet.md), [the C2Net roster](openi_c2net_roster_and_targets.md)).

What that means for a *checkpoint* was living in conversation rather than in the tree: two size lines
that nothing in this repository stated, next to a capacity calculation whose defaults priced a
different machine than the one the project targets. This page writes down both the lines and the two
boxes they are priced against.

## The two size lines

The positioning states itself as two coarse lines, one for each shape of checkpoint:

- **MoE to about 500 B parameters.**
- **Dense to about 200 B parameters.**

They are a statement about which *new* models are worth adapting, and not about what the runtime
supports. The two are not the same shape of limit.

**The dense line is a card-memory limit, and it is the box's number.** A dense model has no part that
can live in host memory, so the whole checkpoint has to be resident, and at the eight-card box below
that is `8 x 32 GiB x 0.85 = 217.6 GiB` of usable card. 200 B parameters at one byte a parameter is
186.26 GiB, which fits with ~4 GiB a card left for a cache and workspace; 200 B at two bytes is
372.5 GiB and fits nothing this project rents. So "dense to 200 B" is a way of saying *eight-bit
weights or smaller, and no bigger than the box*. Half the cards -- the four-card spec the rental
catalog actually offers -- would put the line at 100 B, which is the arithmetic reason the card count
matters more than it looks.

**The MoE line is a host-memory limit.** An MoE keeps its non-expert part on the cards and stages the
routed experts from RAM, so what bounds the size is the bank's reachability, not the cards: a
500 B-parameter MoE at the sparsity of the models here is tens of GiB resident, and hundreds of GiB
of host bank behind it. DeepSeek-V4.1-Flash is the worked example, and it is also why the line is a
rule of thumb rather than a gate -- its backbone is **552 B parameters, above the line**, and it is
the model this repository adapts. Its own page's breakdown is what a parameter count cannot see:
475.24 GiB of checkpoint, 268.95 GiB of it routed experts and 189.13 GiB of it Engram tables,
**16.79 GiB retained** on the cards
([the design record](deepseek_v4_1_flash_design.md), [the model guide](../models/deepseek-v4.1-flash.md)).
457.8 GiB of that bank is pinned host memory when the experts are held resident, which is the budget
that decides whether it runs rather than the 552.

## What is actually on the gate

The gate is gate 0 of the triage ladder, and it is two budgets rather than a size
([the arithmetic](../guides/hardware_triage.md)):

1. the model's **per-rank resident bytes** must fit one card after the reserve, and
2. the **off-card part** of the checkpoint -- routed experts, gather tables -- must fit host memory
   the runtime can actually reach, since that part is staged rather than held.

The second is why an offload plan is not free: when the bank does not fit, the excess comes back onto
the cards and changes the card count rather than being dropped. And what the gate may say is
asymmetric -- a *failure* is definitive, because the arithmetic models no workspace, no fragmentation
and no bank-miss penalty, while a *pass* is provisional for exactly the same reason and routes to
`CANDIDATE` rather than to a tier. `relicllm/triage/fit.py`'s own docstring carries that ruling; it is
repeated here because the positioning is often read as a stronger claim than the check it stands for.

## The two boxes

Two machines appear in this package's arithmetic, and they are not the same machine. Both are named in
`relicllm/components/moe/boxes.py`, which is the single place either one is written down.

| Box | Cards | Card memory | The question it answers |
| --- | ---: | ---: | --- |
| `POSITIONING_BOX`, `8x32gib-positioning` | 8 | 32 GiB | does the hardware we target hold this model |
| `DEV_BOX`, `consumer-gpu-box` | 4 | 22.0 GiB | what the recorded numbers were measured on |

**Eight cards and not four.** The account's 910A catalog tops out at a four-card spec, but that is a
ceiling on what a *rental can ask for*, not on what the card can be run on: the port is measured on
the account's own eight-card 910A machine, and the same eight cards are the reason the dense line
above lands at 200 B. Pricing a fit at four cards prices a model at half the cards it would run on --
and it is the failure direction that hurts, because the card count is what decides whether a model
that runs here comes back as impossible.

**The dev box is the box the numbers came from.** Four cards that report 22,528 MiB, so 22.0 GiB a
card -- and that is the size every threshold here was measured against: Xing4's 17.94 GiB resident
against 18.70 GiB of a 22 GiB card's usable share, and the decode and prefill baselines under
`tests/fixtures/perf`. A fit priced against 32 GiB cards and one priced against 22 GiB cards are both
correct; they answer different questions, which is the reason one of them is a named box rather than
an inherited default.

**The host is unstated on both.** `HardwareProfile.host_memory_gib` defaults to `0`, read as *unsaid*
and never as *none*, so an unstated host turns the offload into a note instead of a silent assumption
that the bank fits.

## Which default is which, and why

`HardwareProfile`'s field defaults and `triage.hardware_profile_for` price the **positioning** box,
because both are answering "would our hardware hold this".

The **dev box** keeps its numbers in the places that describe the machine the code is configuring or
measuring on, where inheriting the targeting box would be wrong about the box in front of the caller:
MiniMax's two plan builders, the `capability_report` ladder (`components/moe/spec.py`'s protocol and
the three implementations behind it), `components/gguf/generation.py`, and `relicllm/cli/triage.py`'s
`--gpu-count` / `--gpu-memory-gib` / `--platform-name`. A triage run against the targeting box is
`--gpu-count 8 --gpu-memory-gib 32` rather than a change of default, because the flag is where a
caller says which machine they mean.

That last one is the reason the split is a default and not a constant: a configuration that silently
re-priced itself onto a machine it is not running on would be a worse bug than the one this page
fixes.
