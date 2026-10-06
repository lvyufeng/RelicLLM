# RelicLLM

Multi-GPU torch runtime for DeepSeek-V4, MiniMax, GLM, Qwen, MiMo and Xing4 checkpoints on older
accelerators — RTX 2080 Ti (`sm_75`) and Ascend 910B. It runs and serves models; it does not own the
kernels. Those are in [relic-core](https://github.com/lvyufeng/relic-core), which both this runtime
and [PocketLLM](https://github.com/lvyufeng/PocketLLM) depend on.

## Language convention

**All Markdown and code comments are written in English.** Commit messages, docstrings and every
`.md` file. A Chinese version of a document is a separate file (`README.md` / `README_CN.md`), never
a mixed-language one.

## One package

**There is one top-level package: `relicllm`.** It was two — `relicllm/` beside a second package
literally named `src/` — inherited from the monorepo's `pocketllm/` + `src/` split. The split was not
a layering claim and not enforced: `relicllm` imported `src` in 8 files and `src` imported `relicllm`
in 1, so they were two packages that called each other. `src` also had no console entry point and no
`__version__`, and it occupied the generic name `src` in `site-packages`. It was merged in.

The two halves are now subpackages, and the old files are where you would look for them:

| Subpackage | Role |
|---|---|
| `api/`, `backends/`, `protocol/`, `server/`, `cli/`, `engine.py`, `supervisor.py` | the **runtime and serving shell** — request lifecycle, HTTP front end, backend adapters, the CLI |
| `models/`, `loader/`, `components/`, `encoding/`, `runtime/` | the **model side** — architectures, weight loaders, per-model kernels' call sites |

`relicllm` is the installed console entry point (`relicllm = "relicllm.cli:main"`, and
`cli/` is a package so that both that entry point and `python -m relicllm.cli.generate_v41` work).
`setup.py` resolves the subpackage list with `find_namespace_packages`.

**The one import that needs care.** `relicllm/__init__.py` eagerly imports `relicllm.api`, so a
module-level import *inside* that chain which resolves back through `relicllm.<something>` runs
against a half-initialized package. Today `relicllm/runtime/device.py` is a leaf (stdlib and torch
only) and `relicllm/runtime/__init__.py` is empty, so the eager chain is safe. Keep it that way, or
make the offending import lazy.

`docs/` is the published site's source (`mkdocs.yml`), 68 pages indexed by `docs/README.md` — model
guides, per-model design records, performance measurements, and migration notes.

**Two things in that tree are build inputs, not pages:** the theme override
(`docs/overrides/main.html`) and the hook (`docs/hooks/llms_txt_staleness.py`). `custom_dir` and
`hooks:` name them, and `exclude_docs` in `mkdocs.yml` is what keeps them off the site. They live
under `docs/` rather than at the repository root because that is the published site's tree and the
only tree the `pages.yml` filter has to know about.

**Those pages still say `PocketLLM`, `pocketllm_*` and `scripts/*.py`.** The rename to `relicllm`
was applied to `CLAUDE.md`, the README and the package, not to the bodies of 347 lines across 52
document pages: a page's prose names the thing it was written about. Rewriting them is tracked as
its own piece of work rather than folded into the move — it is a large diff with no mechanical
answer (`pocketllm_*` metric prefixes and `POCKETLLM_*` variables must *not* be renamed, per
Provenance below), and mixing it into the relocation would bury which pages actually changed.

## What is deliberately not here

- **No kernels.** There is no C++ or CUDA tree in the package. Every op comes through
  `relic_core.kernels.cuda_loader.load_cuda_kernel()` or `relic_core.kernels.ops`. If a change needs
  a kernel edit, it belongs in relic-core, and this repository gets the new binding.
- **No C++ engine.** `cpp_engine/` was retired with the split, and the `cpp` backend that fronted it
  was removed afterwards. The five remaining runtimes are pure PyTorch: `v41`, `mimo`, `xing4`,
  `torch` and `qwen4_exp`.
- **The kernel test suite.** `tests/` here is the model-runtime half; relic-core carries its own.
  The partition is mechanical: a test lives here if it imports anything from `relicllm` beyond
  `relic_core.kernels.*`.

## Install order

**relic-core first**, from its checkout:

```bash
pip install -e ../relic-core --no-build-isolation --no-deps
pip install -e . --no-build-isolation --no-deps
```

`pip install -e .` alone will try to fetch `relic-core` from PyPI, where it is not published.

`--no-deps` matters for the same reason it does in relic-core: the dependency is `torch>=2.0` with
**no upper bound**, resolved from the environment. A pin with a ceiling installs a torch older than
the box runs and rebuilds kernels against the wrong ABI. `requirements.txt` and `pyproject.toml` must
agree.

## Testing

Run from the repository root — no `conftest.py`, no pytest config, so the CWD is what puts `tests` on
`sys.path`:

```bash
python -m pytest tests/ -q
python scripts/check_test_baseline.py     # diff the run against tests/baseline_failures.txt
python -m pytest tests/ -q -rs            # show why each test was skipped
```

`tests/baseline_failures.txt` is a **set of node ids, not a count** — three tests fixed and one
broken is a net improvement in a count and a regression in the tree. An entry means the test *runs
here and fails*; **a skip is not a pass** and belongs nowhere in that file. The skips need a
checkpoint, a card the build did not target, or `/dev/shm` large enough for a resident bank.
`tests/README.md` is the suite's own documentation — read it before changing how the suite is
invoked.

Several modules are also run as scripts directly (`python tests/test_x.py`), which is why their
imports carry an explicit `sys.path` insert and why there is no `conftest.py`.

**No CI runs the suite.** The only workflow here is `.github/workflows/pages.yml`, which builds the
documentation site, so the baseline check is a manual step.

That workflow runs `mkdocs build --strict`, which is the repository's link checker — but only
*within* `docs/`. It also fails when `docs/llms.txt` is stale with the nav, because
`docs/hooks/llms_txt_staleness.py` checks it on every build and `--strict` promotes the warning. So
a nav edit and the regenerated `docs/llms.txt` (`python scripts/gen_llms_txt.py`) belong in the same
commit. Links leaving the repository are absolute URLs the build cannot see; `docs/README.md` says
why they are written that way.

## Git workflow

**Never commit directly to `master`.** Every change goes on a branch and through a pull request.

Branch prefixes: `feature/`, `fix/`, `refactor/`, `docs/`, `perf/` — `<prefix>/<description>`.

Commits: a one-line summary under 72 characters, a blank line, then the explanation starting on
line 3. **Every commit message ends with:**

```
Co-Authored-By: Claude Code <noreply@anthropic.com>
```

PRs: a title under 72 characters, a body covering summary, implementation details and testing
status, **one concern per PR**, and the body ends with:

```
🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

Merged branches are not reliably deleted on `origin`; delete yours locally and remotely.

## These facts are per-host

Paths to checkpoints (`/mnt/data2`, `/mnt/data3`), the CUDA toolkit layout, `/dev/shm` size and the
`POCKETLLM_*` environment variables below are all facts about one development box, not universal
claims. Determine which machine you are on before concluding anything about what can be built, run
or measured — a command that is right on the CUDA box is usually wrong on the Ascend one.

**Environment variables keep the `POCKETLLM_` prefix** (`POCKETLLM_XING4_DIR`,
`POCKETLLM_MIMO_CHECKPOINT`, `POCKETLLM_NCCL_ID_PATH`, …) after the rename to `relicllm`. They are a
contract with existing launch scripts and configs, not module paths — do not "fix" them. The same
goes for `/dev/shm/pocketllm_*_experts*` (bank names that survive between runs) and for
`pocketllm_cpp` (the compiled extension's module name).

## Provenance

Extracted with `git-filter-repo` from the PocketLLM monorepo, history preserved. The `pocketllm`
package was renamed `relicllm`. The `src/` tree arrived alongside it unchanged and was later merged
into `relicllm/`; that merge used `git mv`, so a file's history survives at its new path.