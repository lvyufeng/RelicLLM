# `--config-path` is a runtime profile, and the checkpoint's config is `--checkpoint-config-path`

**Affects:** anyone passing `--config-path` to `relicllm serve`, setting `POCKETLLM_CONFIG_PATH` /
`CONFIG_PATH` / `CONFIG`, or constructing `EngineArgs(config_path=...)`.

## What changed

One flag carried two files that are not the same kind of thing, and they disagree about what a JSON
says:

- a **runtime profile** is what a runtime's own loader expands — the hyperparameters and the
  quantisation selection, in the runtime's spelling (`dim`, `n_layers`, `dtype`, `expert_dtype`);
- a **checkpoint config** is what identification reads to answer *does this runtime serve this
  model* — the release's `config.json`, in the export's spelling (`hidden_size`,
  `num_hidden_layers`, `model_type`).

`torch`'s loader wanted the first; `read_config` in `relicllm/backends/capabilities.py` wanted the
second. One name meant a launch had to leave one of its two questions unanswered, and both ways were
broken:

| Route | What happened |
|---|---|
| the checkpoint's `config.json` under `--config-path` | identification was satisfied, but the loader expanded the checkpoint's spelling — not the profile's key space at all, and with `dtype` defaulting to `fp8` where an fp4 checkpoint expects fp4 |
| an external profile under `--config-path` | the loader was satisfied, but identification read the profile as a checkpoint declaring `model_type=None` and **refused the runtime that had just been configured** |

| | Before | Now |
|---|---|---|
| `--config-path` | either file, and whichever reader needed the other one got a wrong answer | the runtime profile |
| the checkpoint's config | `--config-path`, or `<model>/config.json` by default | `--checkpoint-config-path`, defaulting to `<model>/config.json` as identification always did |
| `EngineArgs.config_path` | the profile *and* the identification input | the profile |
| `EngineArgs.checkpoint_config_path` | did not exist | the identification input |
| a missing default profile | `open("")` → `FileNotFoundError: [Errno 2] No such file or directory: ''` | `ConfigurationError` naming the profile, the detected dtype, and `--config-path` |

## What to do

**A launch that named the profile it needs** is unchanged — `--config-path` still means what it
meant to the loader:

```bash
relicllm serve --model /ckpt --backend torch --config-path /profiles/config_fp4_active.json
```

**A launch that named the checkpoint's config** — move that argument:

```bash
relicllm serve --model /ckpt --backend torch --config-path /ckpt/config.json   # before
relicllm serve --model /ckpt --backend torch --checkpoint-config-path /ckpt/config.json   # now
```

Most launches did not name it at all: `<model>/config.json` was and remains the default, so
`--checkpoint-config-path` is only needed when the architecture lives somewhere else, such as a
GGUF's sidecar.

**The environment bridge** keeps `CONFIG_PATH` / `CONFIG` for the profile, because it was the
loader's input there too. The checkpoint config has no environment spelling; it defaults to
`<model>/config.json`.

## Why it surfaced now, and one launch it unblocks

`torch` refuses to start without a profile it can name, and **no profile ships in this checkout** —
the `configs/` directory `torch_backend` looks in is not in the tree, and never was. A `torch` run
therefore has to be handed one, and naming it is exactly what used to switch identification off. The
committed performance baseline records the path it ran with
(`docs/performance/old_hardware_roadmap.md`), and the profile there is that of the 0731 checkpoint:
its shape keys match the checkpoint exactly (43 layers, 4096 dim, 256 experts, 6 active), with
`dtype: fp8` beside `expert_dtype: fp4`.