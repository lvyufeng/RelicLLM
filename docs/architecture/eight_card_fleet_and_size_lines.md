# The eight-card fleet, and the two lines a checkpoint has to clear

[The OpenI accelerator fleet](openi_accelerator_fleet.md) is what an account can **rent**: 910A up to
four cards, V100 up to two, one task at a time. This page is the other half of the fleet — the
eight-card machines the project runs on directly — and the two size lines that decide whether a
checkpoint is worth downloading before any of them is tried.

No host name and no address is recorded here. Machines are grouped into **classes**, and a class is a
card model, a card count, a host and the budgets that follow from them; a claim about a class is a
claim about every box in it. The card and host figures below were read with `nvidia-smi`,
`npu-smi`, `free -g` and `df -h /dev/shm` on **2026-10-15**, over the same network the runs use.

## The classes

| Class | Cards | Card memory | Host RAM | `/dev/shm` default | Compute | Cores | Boxes |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Eight-way V100 | 8 × V100-PCIE-16GB | 128 GiB | 315 GiB | 158 GiB | `sm_70` | 56 | 3 |
| Eight-way RTX 3090 | 8 × RTX 3090 | 192 GiB | 755 GiB | 378 GiB | `sm_86` | 96 | 1 |
| Eight-way Ascend, 32 GiB | 8 × 32 GiB | 256 GiB | 755 GiB | 378 GiB | 910A / 910B / 910PremiumA | 192 | 12 |
| Eight-way Ascend, 64 GiB | 8 × 910B3 | 512 GiB | 1006 GiB | 504 GiB | 910B3 | 192 | 2 |

Two facts about the table are easy to miss. The `/dev/shm` column is **half of RAM on all four
classes**, because that is the kernel's default `tmpfs` size rather than anything about these boxes;
`mount -o remount,size=` moves it, and the cost of moving it is the RAM the host no longer has for
itself. And the classes differ by more than a factor of three in cores, which is the resource that
decides whether a bank can be *staged* out of, as
[the roadmap](../performance/old_hardware_roadmap.md) found when it diagnosed a served V4.1 step as
host-submission-bound with the device 11% busy.

The 32 GiB Ascend class is one row because `npu-smi` reports one card size and one HBM figure across
the three names, and because the host figures above were read on its **910A** and **910B** members —
the `910PremiumA` boxes are the same card and were not read for this page, so treat their host
numbers as the same shape rather than the same numbers. The class column is the card, not the
generation.

## What the card count decides

A **dense** model has nowhere else to put its bytes: every weight has to be on a card on every step.
Its ceiling therefore scales with the cards, and the fit gate's own `usable_per_card` — 15% held back
for the workspace and the allocator — is the arithmetic:

| Class | 8 × card GiB | Usable at 15% reserve | Dense ceiling at 8-bit |
| --- | ---: | ---: | ---: |
| Eight-way V100 | 128 | 108.8 | ~110 B |
| Eight-way RTX 3090 | 192 | 163.2 | ~160 B |
| Eight-way Ascend, 32 GiB | 256 | 217.6 | ~215 B |
| Eight-way Ascend, 64 GiB | 512 | 435.2 | ~435 B |

A **mixture of experts** is different, and this is the correction that moved this page. At TP8 the
resident part is the attention stack, the embeddings, the norms and the shared experts; the routed
experts and the gather tables live in a host bank. DeepSeek-V4.1-Flash's 475.24 GiB checkpoint is
457.78 GiB of bank, so **17.5 GiB is resident** — under 3 GiB a card over eight ranks, before the KV
cache. Above a handful of cards the card count stops being the binding constraint for a MoE at all,
and the ceiling lands on the host.

Eight ranks is a configuration, not a code change. `ShardPlan.build(cfg, rank, 8)` accepts the
released geometry, `o_groups` is 8 and `groups / world` is 1, which is what
`tests/test_models_deepseek_v4_1_tp.py` pins against the released config. The reason to plan an
eight-card box at eight cards is that the resident share and the KV are the two things that divide;
the reason it buys nothing for a MoE is the same sentence read the other way, since the bank does
not divide.

## The MoE line is a host budget

The bank's size is a property of the artifact, and the roster's own measurements are the only honest
anchor for it:

| Checkpoint | Params | Bytes on disk | What those bytes are |
| --- | ---: | ---: | --- |
| MiMo-V2.6-Flash | ~310 B | 149.81 GiB | the routed-expert bank itself, MXFP4 |
| GLM-5.3-Flash | 320.76 B | 101.24 GiB | the whole checkpoint, so an upper bound on its bank |
| DeepSeek-V4.1-Flash | 552 B | 475.24 GiB | 457.78 GiB offloadable (268.95 experts + 188.83 Engram) |
| MiniMax-M2.7 | ~230 B | `UD-IQ1_M` | the whole checkpoint, 1–2 bits a weight |

At 4 bits a 300 B-class MoE is therefore 0.3–0.5 GiB per billion parameters, and the line follows the
host rather than the cards:

| Class | Bank budget at the default mount | Roughly, at 4 bits |
| --- | ---: | ---: |
| Eight-way V100 | 158 GiB | ~300 B — MiMo's 149.81 GiB is right at the edge |
| Eight-way RTX 3090 | 378 GiB | ~750 B |
| Eight-way Ascend, 32 GiB | 378 GiB | ~750 B |
| Eight-way Ascend, 64 GiB | 504 GiB | ~1000 B |

That is where the "MoE up to 500 B" rule comes from and what is wrong with stating it that way: 500 B
is a conservative read of the 378 GiB class, the class most of the fleet is in, and the number that
actually governs is the host's bank budget. The parameter figure is the 4-bit translation of a byte
budget, and it moves when the artifact does: the same class holds a 1000 B MoE at 2 bits and refuses
a 400 B one that left its experts at 8.

The V100 class is the one place the line binds in a way card count cannot fix. Its host has 315 GiB
of RAM, so the 491.5 GB that V4.1's bank needs has nowhere to be, at any mount size and at any number
of cards. An eight-card box is not a licence to run an eight-card model; it is a licence to run
8 × 16 GiB of resident weights next to whatever the host can hold.

## The exception, by name

**DeepSeek-V4.1-Flash stays on the list at 552 B.** It is the checkpoint this runtime was extracted
from, the only 262,144-token path that has been served end to end, and the run behind the roadmap's
decode diagnosis. Its bank fits exactly the eight-way 64 GiB Ascend class at the default mount, and
the 755 GiB classes once `/dev/shm` is remounted above 468 GiB and the host keeps ~290 GiB for itself.
An exception written down by name costs one line; moving the line to 552 B would let everything under
it through, which is the opposite of what the line is for.

## Active parameters are a third column

Byte totals set whether a checkpoint runs, not how fast, and two 300 B MoEs with the same bank can
be an order of magnitude apart:

| Checkpoint | Active params | Decode |
| --- | ---: | ---: |
| DeepSeek-V4.1-Flash | 16 B | 3.48–3.54 tok/s |
| MiMo-V2.6-Flash | 15 B | 6.40 tok/s short-context, 8.53 with experts resident |
| Qwen3.8-Flash-Next | 3 B over 128 experts | 3.6 tok/s, PCIe-bound, 28.6 ceiling |

The paragraph to keep in mind when a line is being drawn is the third row: 3 B of active parameters
is the smallest of the three and the slowest of the three, because the experts cross the bus. A rule
stated in total parameters reads this model as cheap.

## What the lines do not cover

Three gates sit outside them, and each one has stopped a candidate that cleared both lines:

- **Gencode.** The fatbin carries `7.5;8.9` SASS, so the V100 class's `sm_70` and the 3090 class's
  `sm_86` are both outside it. The
  [C2Net roster](openi_c2net_roster_and_targets.md) already tracks the `sm_70` gap and the
  `__CUDA_ARCH__ >= 750` guards that compile Volta out of some kernels.
- **The Ascend port.** The Ascend classes have CANN but no `torch_npu` installed, and the Ascend side
  of this tree is still the [device plane](device_plane.md)'s decision surface rather than a set of
  operators. Both Ascend classes are blocked on a port, not on arithmetic.
- **The rentable pool's own ceilings.** An OpenI rental tops out at four 910A cards and two V100
  cards with one task at a time, so the eight-card classes above are the ones run directly. The two
  pools need two sets of lines, and this page is only about the second.

## Where the lines sit

They are a **download and triage pre-filter**, not a fit verdict. Gate 0 already answers fit with
`weights + kv + workspace ≤ usable_per_card`, and it treats a host bank as placed rather than
unplaced; where the bank is larger than host memory the excess comes back onto the cards and moves
the card count. So a line that is too loose costs one download and one disk slot, and a line that is
too tight throws away a model that would have run — which is the direction worth erring in.
