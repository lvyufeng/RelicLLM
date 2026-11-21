"""Explicit compact expert format for the unchanged full-checkpoint trial chain."""
from __future__ import annotations

import inspect
from pathlib import Path
import profile_expert_tp as base


def main():
    source = inspect.getsource(base.main)
    marker = "'            tp_world=int(os.environ[\"RELIC_EXPERT_TP\"]), capture_graphs=True, restore_residency=False).install()\\n'"
    replacement = "'            tp_world=int(os.environ[\"RELIC_EXPERT_TP\"]), capture_graphs=True, restore_residency=False,\\n'\n        '            bank_format=os.environ[\"RELIC_COMPACT_BANK_FORMAT\"]).install()\\n'"
    if source.count(marker) != 1:
        raise RuntimeError('the explicit TP format-install boundary moved')
    source = source.replace(marker, replacement, 1)
    source = source.replace('original_expert_tensor_parallel_tile_consumer', 'explicit_compact_or_original_expert_tensor_parallel_consumer', 1)
    source = source.replace('profile-expert-tp-generated-rank-', 'profile-compact-tp-generated-rank-', 1)
    generated = Path(__file__).with_name(f"profile_compact_tp_driver-rank-{base.os.environ['LOCAL_RANK']}.generated.py")
    # Every worker keeps an exclusive immutable source for inspect.getsource.
    if not generated.exists():
        try:
            with generated.open('x') as handle: handle.write(source)
        except FileExistsError:
            pass
    if generated.read_text() != source:
        raise RuntimeError('the frozen compact profile driver differs')
    exec(compile(source, str(generated), 'exec'), base.__dict__)
    return base.main()


if __name__ == '__main__':
    raise SystemExit(main())
