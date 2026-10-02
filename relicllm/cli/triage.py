"""``python -m relicllm.cli.triage`` — the two-phase front end.

Standalone rather than a subcommand of ``relicllm.cli``, matching the ``inspect_gguf.py`` precedent:
``relicllm/cli/__init__.py:239`` has ``if args.command != "serve": return 2``, so registering a
second subcommand there would mean editing the dispatcher and its tests to gain nothing this module
does not already have.

Two phases, and the split is the design. Phase one is free -- header arithmetic, no cards, no
allocation -- so it runs unconditionally and prints the command that would settle the question.
Phase two happens only after a human has decided the GPU-hours are worth it, and its input is a
results file rather than a flag, so the measurement has to have come from somewhere nameable.

Exit codes are the tool's contract with a script: ``0`` for a tier at or above demo, ``1`` for
impossible, ``2`` for a usage error, ``3`` for a checkpoint that could not be read. A caller that only
wants to know "is this worth looking at" reads ``$?`` and never parses the table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from relicllm.triage.checkpoint import checkpoint_paths
from relicllm.triage.fit import DEFAULT_BATCHES, DEFAULT_CONTEXTS, hardware_profile_for
from relicllm.triage.report import Assessment, assess, dumps
from relicllm.triage.tier import DEFAULT_THRESHOLDS, Measured, Thresholds, Tier

__all__ = ["EXIT_OK", "EXIT_IMPOSSIBLE", "EXIT_UNREADABLE", "EXIT_USAGE", "build_parser", "main"]


EXIT_OK = 0
EXIT_IMPOSSIBLE = 1
EXIT_USAGE = 2
EXIT_UNREADABLE = 3

TIER_EXIT_CODES = {
    Tier.PRODUCTION: EXIT_OK,
    Tier.CHAT: EXIT_OK,
    Tier.DEMO: EXIT_OK,
    Tier.CANDIDATE: EXIT_OK,
    # The one tier that is a refusal rather than a ranking, and therefore the one a script branches on.
    Tier.IMPOSSIBLE: EXIT_IMPOSSIBLE,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m relicllm.cli.triage",
        description=(
            "Decide whether a checkpoint is worth adapting to this hardware, and to what service "
            "level: production, chat, demo, or not at all."
        ),
    )
    parser.add_argument(
        "checkpoint",
        nargs="+",
        help="checkpoint directory, .gguf file, or a directory of GGUF quant subdirectories",
    )
    parser.add_argument("--json", action="store_true", help="emit the machine-readable report")
    parser.add_argument(
        "--brief",
        action="store_true",
        help="print the tier and the one-line reason only, for a list of checkpoints",
    )

    box = parser.add_argument_group("the box")
    box.add_argument("--gpu-count", type=int, default=4)
    box.add_argument("--gpu-memory-gib", type=float, default=22.0)
    box.add_argument(
        "--host-memory-gib",
        type=float,
        default=0.0,
        help=(
            "RAM the offload bank may use. Unset means *unsaid*, which turns the offload into a "
            "note rather than assuming the bank fits"
        ),
    )
    box.add_argument("--platform-name", default="consumer-gpu-box")
    box.add_argument(
        "--reserve-fraction",
        type=float,
        default=0.15,
        help=(
            "fraction of each card held back from the fit. The two values in use differ by 1.1 GiB "
            "on a 22 GiB card, which decides whether Xing4 has 0.65 or 1.86 GiB of headroom"
        ),
    )
    box.add_argument(
        "--host-reserve-fraction",
        type=float,
        default=0.10,
        help="fraction of host memory held back from the offload bank",
    )
    box.add_argument("--workspace-bytes", type=int, default=0, help="per-card workspace to subtract")

    shape = parser.add_argument_group("the shape to solve for")
    shape.add_argument("--context", type=int, default=8192, help="context length to price the card count at")
    shape.add_argument("--batch", type=int, default=1, help="batch width to price the card count at")
    shape.add_argument(
        "--contexts", type=int, nargs="+", default=list(DEFAULT_CONTEXTS), help="the curve's context grid"
    )
    shape.add_argument(
        "--batches", type=int, nargs="+", default=list(DEFAULT_BATCHES), help="the curve's batch grid"
    )
    shape.add_argument(
        "--no-siblings",
        action="store_true",
        help="do not look for other precisions of the same model beside this one",
    )

    thresholds = parser.add_argument_group("the SLOs")
    thresholds.add_argument("--ttft-1k-seconds", type=float, default=DEFAULT_THRESHOLDS.ttft_1k_seconds)
    thresholds.add_argument("--tpot-seconds", type=float, default=DEFAULT_THRESHOLDS.tpot_seconds)
    thresholds.add_argument(
        "--prefill-8k-tok-per-second",
        type=float,
        default=DEFAULT_THRESHOLDS.prefill_8k_tok_per_second,
    )

    measured = parser.add_argument_group("phase two")
    measured.add_argument(
        "--measured",
        metavar="RESULTS.JSON",
        help=(
            "a JSON file of measurements from a run; '-' reads stdin. Re-run without it to get "
            "CANDIDATE and the command that would produce this file"
        ),
    )
    return parser


def _thresholds_from(args: argparse.Namespace) -> Thresholds:
    return Thresholds(
        ttft_1k_seconds=args.ttft_1k_seconds,
        tpot_seconds=args.tpot_seconds,
        prefill_8k_tok_per_second=args.prefill_8k_tok_per_second,
        input_len=DEFAULT_THRESHOLDS.input_len,
        output_len=DEFAULT_THRESHOLDS.output_len,
    )


_MEASURED_FIELDS = (
    "ttft_1k_seconds",
    "tpot_seconds",
    "prefill_8k_tok_per_second",
    "ttft_1k_p99_seconds",
    "tpot_p99_seconds",
    "source",
)


def _measured_from(path: str) -> Measured:
    """Read a results file, accepting either this tool's own shape or a flat mapping of the fields.

    Deliberately tolerant about *which* keys are present, because a partial run is the normal case --
    a latency sweep produces TTFT and TPOT and no prefill number, and refusing the file would push
    the caller toward inventing the missing field. Missing fields stay missing and
    :func:`~relicllm.triage.tier.tier` reports them as the reason for a ``CANDIDATE``.
    """
    raw = sys.stdin.read() if path == "-" else Path(path).read_text()
    payload: Any = json.loads(raw)
    if isinstance(payload, dict) and "measured" in payload and isinstance(payload["measured"], dict):
        payload = payload["measured"]
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not hold a JSON object of measurements")
    fields = {key: payload[key] for key in _MEASURED_FIELDS if key in payload}
    if not fields:
        raise ValueError(
            f"{path} holds none of {', '.join(_MEASURED_FIELDS)}; "
            "a file with no recognised measurement would silently read as an empty run"
        )
    return Measured(**fields)


def _assessment_for(args: argparse.Namespace, checkpoint: str, measured: Measured | None) -> Assessment:
    return assess(
        checkpoint,
        hardware=hardware_profile_for(
            gpu_count=args.gpu_count,
            gpu_memory_gib=args.gpu_memory_gib,
            host_memory_gib=args.host_memory_gib,
            name=args.platform_name,
        ),
        measured=measured,
        thresholds=_thresholds_from(args),
        context=args.context,
        batch=args.batch,
        reserve_fraction=args.reserve_fraction,
        host_reserve_fraction=args.host_reserve_fraction,
        workspace_bytes=args.workspace_bytes,
        contexts=tuple(args.contexts),
        batches=tuple(args.batches),
        include_siblings=not args.no_siblings,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    measured: Measured | None = None
    if args.measured:
        try:
            measured = _measured_from(args.measured)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(f"triage: --measured {args.measured}: {error}", file=sys.stderr)
            return EXIT_USAGE

    reports: list[Assessment] = []
    for checkpoint in args.checkpoint:
        # One argument can name several releases. The help text has always promised "a directory of
        # GGUF quant subdirectories", and `GLM-5.2-GGUF/` holding `UD-Q2_K_XL/` beside `UD-Q4_K_M/`
        # is what that looks like on disk -- the whole point of pointing at the parent is to compare
        # them. `checkpoint_paths` is the loader's own answer to what a path holds, so it answers
        # here too rather than a second guess being written next to it.
        try:
            releases = checkpoint_paths(checkpoint)
        except OSError:
            releases = (checkpoint,)
        # One release is not a choice, so the path stands as the caller wrote it and the report keeps
        # naming what they pointed at. That matters for a sharded upload, where the directory is the
        # model and the shard files are not: expanding those would relabel every row after a part.
        if len(releases) == 1:
            releases = (checkpoint,)
        for release in releases:
            try:
                reports.append(_assessment_for(args, release, measured))
            except Exception as error:  # a bad path must not stop the other checkpoints
                print(f"triage: {release}: {type(error).__name__}: {error}", file=sys.stderr)
                return EXIT_USAGE

    if args.json:
        payload = reports[0].to_dict() if len(reports) == 1 else [report.to_dict() for report in reports]
        print(json.dumps(payload, indent=2))
    elif args.brief:
        for report in reports:
            name = Path(report.checkpoint).name
            # A brief row that printed CANDIDATE for a checkpoint whose headers did not read would
            # be the exact conflation the report's own UNREADABLE state exists to prevent: a
            # filesystem error is not a claim about the model, and the exit code says so already.
            if report.unreadable:
                print(f"{'UNREADABLE':11s} {name}  --  {report.unreadable}")
                continue
            print(f"{str(report.verdict.tier).upper():11s} {name}  --  {report.verdict.reason}")
    else:
        for index, report in enumerate(reports):
            if index:
                print()
            print(report.format_text())

    exit_code = EXIT_OK
    for report in reports:
        if report.unreadable:
            return EXIT_UNREADABLE
        if TIER_EXIT_CODES[report.verdict.tier] != EXIT_OK:
            exit_code = TIER_EXIT_CODES[report.verdict.tier]
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
