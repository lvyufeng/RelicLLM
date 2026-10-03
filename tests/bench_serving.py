#!/usr/bin/env python3
"""The vLLM-convention serving benchmark, kept as a script under ``tests/``.

The implementation moved into the package -- ``relicllm.bench.client`` -- so that ``relicllm bench``
can use it and so the definitions the metrics rest on live next to the code rather than beside it. This
module is the compatibility shim: it re-exports every name the test and the other scripts import, and
its ``main()`` forwards to the same client parser, so ``python tests/bench_serving.py …`` still works.

The one thing that did **not** move is the rule this file used to carry in its docstring: **the client
never launches a server.** It measures a base URL. The launch is now ``relicllm bench serve``'s job
(``relicllm/bench/launcher.py``), which is a separate concern in a separate module -- and this shim
still has no ``--ckpt``, ``--binary`` or ``--devices`` to fall back on, which
``tests/test_bench_serving_metrics.py`` checks.

Run it as a script:
    python tests/bench_serving.py --base-url http://127.0.0.1:8123 --model Qwen3.8-27B \
        --random-input-len 128 --random-output-len 32 --num-prompts 16 \
        --request-rate 2 --goodput ttft:2000 tpot:60 --json-out /tmp/serve.json
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np  # noqa: F401 - the test reaches `bench_serving.np`

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import relicllm.bench.client as _client  # noqa: E402

# Re-exported verbatim: the test loads this module by path and reaches these names on it.
from relicllm.bench.client import (  # noqa: E402,F401
    DEFAULT_BASE_URL,
    DEFAULT_PERCENTILE_METRICS,
    GOODPUT_KEYS,
    MILLISECONDS_TO_SECONDS,
    HttpResult,
    RequestOutput,
    SampleRequest,
    apply_tokenizer_counts,
    arrival_delays,
    build_dataset,
    build_payload,
    calculate_metrics,
    discover_model,
    http_request,
    nonstream_request,
    parse_goodput,
    parse_percentiles,
    print_summary,
    prompt_text,
    require,
    run_one,
    selected_percentile_metrics,
    stream_request,
    warm_up,
)

#: The name ``--json-out``'s ``script`` field used to record. Kept so a record written by this shim is
#: still identifiable as having come from the old entry point; the package's own record carries
#: ``tool: "relicllm bench"`` instead.
SCRIPT_NAME = "tests/bench_serving.py"


def build_parser():
    """The client parser, with ``--backend`` kept for a command line that still passes it."""
    parser = _client.build_parser()
    parser.add_argument("--backend", default="openai", choices=["openai"], help="Serving backend to benchmark (kept for compatibility).")
    return parser


def run(args):
    """Measure the server and, with ``--json-out``, write the record — as the script always did."""
    record = _client.run_client(args)
    record["script"] = SCRIPT_NAME
    record["git_commit"] = _git_head()
    if args.json_out:
        import json  # noqa: PLC0415 - only needed to write

        import pathlib  # noqa: PLC0415

        pathlib.Path(args.json_out).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def _git_head() -> str:
    import subprocess  # noqa: PLC0415

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> int:
    # ``build_parser()`` here adds ``--backend`` on top of the client's, so it parses a superset; the
    # client's ``run_client`` only reads the fields it knows.
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())