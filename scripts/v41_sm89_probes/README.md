# sm_89 (RTX 4090) V4.1 probes

Read-only probes for the 8x RTX 4090 box. Nothing here is part of the runtime and nothing here
touches model weights or the checkpoint.

Run with the relic-core checkout and the RelicLLM tree on `PYTHONPATH`:

```bash
PYTHONPATH=<relic-core>:<RelicLLM> python probe.py
```

`-I` is deliberately **not** used here, unlike the [sm_75 probes](../v41_sm75_probes/README.md). On
the 4090 boxes an editable install resolves `import relicllm` to a different scratch tree, so the
`PYTHONPATH` is what selects the tree under test and isolated mode would drop it.

| Script | What it establishes |
|---|---|
| `probe.py` | Where a decode step goes under the MMA emulation: `mma.fp8_block_gemm` at the layer's own `wq_a` shape (M=1 decode and M=512 prefill, both scale-grid layouts) against the correctly-rounded `F.linear` it replaces, plus the `block_fp8_act_quant` it gates on. The emulation's cost has always been op count rather than FLOPs, so the ratio to a plain GEMM is the number that says how much of a step it is. |

These are the probes the plan's Cell 4 rests on: the criterion configuration is too slow to bench at
128 output tokens, so the step has to be attributed rather than timed end to end. A GEMM's speed is
not a token-level claim, and nothing here makes one.