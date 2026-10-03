"""``relicllm bench serve`` — launch a server (or target a running one) and measure it.

Named after ``vllm bench serve``, and for the same reason that command exists: the measurement is a
first-class verb, not a script beside the server. Bench owns its own flags; the command line that
starts the server goes **after ``--``**, verbatim::

    relicllm bench serve --scenario decode --json-out /tmp/bench.json \
        -- --model /path/to/ckpt --backend auto --tensor-parallel-size 4

    relicllm bench serve --base-url http://127.0.0.1:8123 --scenario prefill-8k

The separator is the whole design. Re-declaring ``serve``'s flags here would be a second copy of a
surface that already drifts by omission -- a flag added to ``serve`` and not here is a flag the
benchmark silently cannot pass -- so instead the tail after ``--`` is handed to the child unchanged,
and ``relicllm serve``'s own parser is the only thing that reads it. A flag added to ``serve`` is
measurable the day it lands, with no change here.

With ``--base-url`` there is nothing to launch, which is the second half of the same design: the
numbers a client observes and the process that produced them are separable, and a server started by
whoever is tuning it is measured by pointing at it. The record says which of the two happened.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Sequence

from relicllm.bench import client, scenarios
from relicllm.bench.launcher import DEFAULT_READY_TIMEOUT, LaunchError, launch
from relicllm.bench.metadata import dumps, envelope

__all__ = ["EXIT_OK", "EXIT_USAGE", "add_arguments", "add_subparser", "build_parser", "main", "run"]

EXIT_OK = 0
EXIT_USAGE = 2


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The ``bench serve`` flags: how to reach a server, then what to run against it."""
    reach = parser.add_argument_group("the server to measure")
    reach.add_argument(
        "serve_argv",
        nargs="*",
        help="the serve command, after `--`: `relicllm bench serve -- --model /ckpt --backend auto ...`",
    )
    reach.add_argument("--host", default="127.0.0.1", help="interface the launched server binds (default: 127.0.0.1)")
    reach.add_argument("--port", type=int, default=0, help="port the launched server binds; 0 picks a free one")
    reach.add_argument(
        "--ready-timeout",
        type=float,
        default=DEFAULT_READY_TIMEOUT,
        metavar="SECONDS",
        help=(
            "how long to wait for /ready after launching. Sized to the checkpoint, not to the model: a "
            "cold host-expert preload is thousands of seconds a rank (default: 1800)"
        ),
    )
    reach.add_argument("--launch-log-dir", default=None, help="directory for the launched server's log (default: cwd)")
    reach.add_argument("--keep-server", action="store_true", help="leave the launched server running when the run ends")

    work = parser.add_argument_group("what to run")
    work.add_argument(
        "--scenario",
        action="append",
        choices=scenarios.names(),
        default=None,
        metavar="NAME",
        help=(
            "a named workload from relicllm/bench/scenarios.py, repeatable. One of: "
            + ", ".join(scenarios.names())
            + " (default: decode)"
        ),
    )
    work.add_argument("--json-out", metavar="PATH", help="write the full record (metadata + one entry per scenario) here")

    # `suppress_defaults=True` keeps a flag the operator did not name off the namespace, so `run` can
    # tell "the operator said 2" from "the scenario's default is 2". Without it an explicit
    # `--num-prompts 2` would be overwritten by the scenario and silently ignored.
    client.add_client_arguments(parser, suppress_defaults=True)


def add_subparser(subparsers: Any) -> argparse.ArgumentParser:
    """Register ``bench`` on the top-level parser so ``relicllm --help`` lists it."""
    bench_parser = subparsers.add_parser("bench", help="measure a served model on vLLM's terms")
    bench_subparsers = bench_parser.add_subparsers(dest="bench_command", required=True)
    serve_parser = bench_subparsers.add_parser("serve", help="launch (or target) a server and measure it")
    add_arguments(serve_parser)
    return bench_parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="relicllm bench",
        description="Measure a served model on vLLM's terms: TTFT, TPOT, ITL, E2EL, throughput, goodput.",
    )
    subparsers = parser.add_subparsers(dest="bench_command", required=True)
    serve_parser = subparsers.add_parser("serve", help="launch (or target) a server and measure it")
    add_arguments(serve_parser)
    return parser


def _split_serve_tail(argv: Sequence[str]) -> tuple[list[str], list[str] | None]:
    """Split ``argv`` at the first bare ``--`` into (bench flags, serve command).

    The tail is returned as a separate list rather than as a positional so argparse never sees it: a
    serve flag that happened to collide with a bench flag, or a value starting with ``-``, would
    otherwise be parsed by the wrong parser. ``None`` means no ``--`` was present, which is not the
    same as an empty tail -- the first is "no server named", the second is "-- with nothing after it".
    """
    for index, item in enumerate(argv):
        if item == "--":
            return list(argv[:index]), list(argv[index + 1 :])
    return list(argv), None


def _serve_tail(argv: Sequence[str]) -> list[str]:
    return [item for item in argv if item != "--"]


def _client_namespace(args: argparse.Namespace, name: str, base_url: str) -> argparse.Namespace:
    """The client's view of one run: its own flags, the resolved URL, and the scenario.

    Order is the whole point. Client defaults first, then the scenario (which must supply a value for
    every workload field the operator did not name), then the operator's own flags on top -- so an
    explicit ``--num-prompts 2`` beats the scenario's count rather than being overwritten by it.
    ``explicit`` is what the operator's flags are: the parsed namespace carries only what was named,
    because the bench parser suppressed the client defaults.
    """
    namespace = scenarios.default_client_args()
    scenarios.apply(scenarios.resolve(name), namespace)
    explicit = {key for key in vars(args) if hasattr(namespace, key)}
    for key in explicit:
        setattr(namespace, key, vars(args)[key])
    namespace.base_url = base_url
    return namespace


def run(args: argparse.Namespace, serve_argv: list[str] | None = None) -> int:
    """Launch or target a server, measure each scenario, write the record, tear down."""
    if serve_argv is None:
        serve_argv = _serve_tail(getattr(args, "serve_argv", []) or [])

    if args.base_url and serve_argv:
        print(
            "relicllm bench: --base-url measures a server somebody else started, so the command after "
            "`--` has nothing to start; pass one or the other",
            file=sys.stderr,
        )
        return EXIT_USAGE

    if not args.base_url and "--model" not in serve_argv:
        print(
            "relicllm bench: nothing to launch. Name the server after `--` "
            "(relicllm bench serve -- --model /path/to/ckpt ...) or point at one with --base-url",
            file=sys.stderr,
        )
        return EXIT_USAGE

    selected = args.scenario or ["decode"]
    server = None
    launch_record: dict[str, Any] | None = None
    try:
        if args.base_url:
            base_url = args.base_url.rstrip("/")
        else:
            server = launch(
                serve_argv,
                host=args.host,
                port=args.port,
                ready_timeout=args.ready_timeout,
                log_dir=args.launch_log_dir,
            )
            base_url = server.base_url
            launch_record = {"argv": server.command, "log": str(server.log_path)}

        records: dict[str, Any] = {}
        for name in selected:
            records[name] = client.run_client(_client_namespace(args, name, base_url))

        record = envelope(
            host="bench",
            base_url=base_url,
            launch=launch_record,
            scenarios=records,
        )
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as handle:
                handle.write(dumps(record))
    except LaunchError as error:
        print(f"relicllm bench: {error}", file=sys.stderr)
        return EXIT_USAGE
    finally:
        if server is not None and not args.keep_server:
            server.stop()

    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    head, tail = _split_serve_tail(argv)
    args = build_parser().parse_args(head)
    return run(args, serve_argv=tail)


if __name__ == "__main__":
    raise SystemExit(main())