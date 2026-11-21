# Where the Ascend decode path can be intercepted

This page exists because of a repeated mistake. Across three rounds an optimisation was
implemented here that the tree already had: a widened-weight cache that `cube_weight_cache.py`
already provided, an attention-dispatch hoist that `sparse_attn` already did, and a batch-one
routing fast path that `M1RoutedDispatch` already did. Each was correct and each was wasted, because
the check that would have found the existing implementation — one grep for the concept — was skipped.

The second half of that check matters more than the first. **A correct change in the wrong seam is
worth nothing.** The candidate configuration runs a stack of driver-installed substitutions, so the
obvious implementation of a step is often not the one that executes. This page records the seams,
what patches each, and which configuration selects it, so that the next change starts by knowing
where it will actually land.

## The seams, outermost first

| seam | patched by | selected by | what it replaces |
|---|---|---|---|
| the whole decode forward | `ascend_graphs.NpuDecodeGraphs` | constructed, not flagged | a round of recorded decode steps, one `NpuLayerGraphs` per block |
| host Engram work | `host_engram_prefetch.HostEngramPrefetch` | `RELIC_HOST_ENGRAM_PREFETCH=1` | the per-token host lookups, moved ahead of the device work |
| activation quantize and decode | `fused_activation_binding.FusedActivationBinding` | `RELIC_FUSED_ORIGINAL_ACTIVATION=1` | **relic-core module functions** `block_fp8_quant.quantize_block_fp8` and `unpack.unpack_fp8_to_bf16` |
| expert dispatch | `ascend_m1_routing.M1RoutedDispatch` | `RELIC_M1_DIRECT_ROUTING=1` | **`store.forward`** on every routed store |
| expert body | `ascend_expert_graphs.RoutedExpertGraphs` | `RELIC_EXPERT_BODY_GRAPHS=1` | the **module attribute** `ascend_packed_experts.packed_expert_forward` |
| device-selected staged MoE | `ascend_device_routing.DeviceRoutedDispatch` | explicit constructor; private driver `RELIC_DEVICE_MOE=1` | **`store.forward`**, original row-major bank, shared selected I8/BF16 stage |
| device-selected tile MoE | `ascend_tile_routing.TileRoutedDispatch` | explicit constructor; private driver `RELIC_DEVICE_TILE_MOE=1` | **`store.forward`**, existing lossless tile bank, bounded native AIV/GM-ring/AIC consumer |

## Why the two expert seams compose rather than shadow

They patch different objects, and the inner one is looked up late on purpose.
`M1RoutedDispatch.install` replaces each store's `forward`. Its replacement does
`from . import ascend_packed_experts as packed` and then calls `packed.packed_expert_forward(...)`
through the module attribute, with the comment *"Resolve the module seam at the call"*.
`RoutedExpertGraphs.install` sets that module attribute. So the M1 path reaches the recorded bodies,
and both bindings are live at once.

This is also why a change inside the stores' bucketed `forward` cannot affect the candidate: with
`RELIC_M1_DIRECT_ROUTING=1` that method is not called for decode at all. The M1 replacement handles
the single-row case itself and delegates only when the input is not a one-row NPU tensor.

## The three families of interception, and what each can and cannot see

The seams fall into three kinds, and which kind a change belongs to decides whether it can work at
all:

1. **Module-function seams** (`quantize_block_fp8`, `unpack_fp8_to_bf16`,
   `packed_expert_forward`). A substitution here is seen by every caller that resolves the name at
   call time, which is why the graph driver and the activation binding reach deep code with a single
   assignment. A caller that did `from x import f` at import time will *not* see it — that is the trap
   in this family.
2. **Method seams** (`store.forward`). Straightforward to substitute, but a later substitution of the
   same method wins, and a caller that bypasses the method is unaffected. Order in the driver is
   therefore load-bearing: the activation binding installs before the M1 dispatch, which installs
   before the expert-graph round.
3. **Construction seams** (`NpuDecodeGraphs`). Not flag-selected; the driver constructs one, and that
   construction is the opt-in. Nothing downstream can be substituted into an already-recorded graph,
   which is why the device kernel profile is identical across changes made after capture.

## The practical consequence

A change to the *Python* path before capture cannot change a recorded graph's device work. The
kernel profile taken through a recorded round is therefore stable no matter what the dispatch code
does, and the round-34 measurement that came back bit-identical — 13,827 kernels per step, 125
`IndexSelect+Gather` calls — was reporting a recorded body, not the code that had just been edited.

So the check before writing an optimisation is two questions, not one:

- does the tree already implement this concept? (`grep` for it)
- which seam does the active configuration use, and does my change land in that seam?

## The opt-in fused/device-routed alternatives

The device-routed dispatcher replaces the same store method as the M1 host dispatcher and therefore
must not coexist with it. It reads original arena references, shares one stage workspace across
layers, and captures device-selected decode bodies; it does not call the module expert-body seam.
All routed IDs/counts stay on device. Prefill delegates to the original store and complete Engram
stays on CPU. The implementation requires one serialized execution stream.

The expert epilogue binding is a separate module-function seam. It installs **before** the expert-body
graph owner so captured bodies contain the fused prologue/encoding rather than an after-capture
Python substitution. Close the outer graph owner first. Original FP8 code/scale outputs retain their
same-kernel decoded BF16 owner through the activation-binding identity guards. Neither import nor
installation replaces CPU Engram's provider.

Do not confuse the two experiments: an epilogue-only pilot retains host routing; the device-routed
body removes routing readback but has different preparation/SDK costs. Their full-model timings,
not the existence of fusion, decide whether either is an optimization.
