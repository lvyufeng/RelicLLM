"""Hermetic tests for ``relicllm bench`` — the orchestration, without a GPU or a checkpoint.

The measurement arithmetic is pinned by ``tests/test_bench_serving_metrics.py``; what is new here is
the *orchestration*: that the command launches a server, waits for readiness, measures it, records what
it launched, and tears it down. None of that needs a real model — the launcher's ``spawn`` seam takes a
process, and a thread that speaks just enough HTTP stands in for the server.

The one end-to-end case runs a real child process (a tiny Python program) so the poll/exit/timeout
paths are exercised against a real ``Popen`` rather than a mock.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from relicllm.bench import client, scenarios
from relicllm.bench import launcher as launcher_module
from relicllm.bench.launcher import LaunchError, launch
from relicllm.cli import bench as bench_cli
from relicllm.cli import main as cli_main


# ---------------------------------------------------------------------------
# A stub server: /ready, /v1/models, and a streamed completion
# ---------------------------------------------------------------------------


class _StubHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/ready":
            if self.server.probes < self.server.not_ready_probes:
                self.server.probes += 1
                self._json(503, {"ready": False})
                return
            self._json(200, {"ready": True})
        elif self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": "stub-model", "object": "model"}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        role = {"id": "c", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
        self.wfile.write(b"data: " + json.dumps(role).encode() + b"\n\n")
        self.wfile.flush()
        for _ in range(max(1, int(body.get("max_tokens", 2)))):
            chunk = {"id": "c", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": None}]}
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            self.wfile.flush()
        done = {"id": "c", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        self.wfile.write(b"data: " + json.dumps(done).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep the test output clean
        pass


class _StubServer:
    def __init__(self, *, not_ready_probes: int = 0):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
        self.server.probes = 0
        self.server.not_ready_probes = not_ready_probes
        self.thread = __import__("threading").Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}"

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5.0)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()
        return False


def _ready_process() -> subprocess.Popen:
    """A real child that stays up, standing in for a server the launcher spawned."""
    program = "import time\nimport sys\nsys.stdout.write('ready\\n')\nsys.stdout.flush()\ntime.sleep(60)\n"
    return subprocess.Popen(
        [sys.executable, "-c", program],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _stub_port(stub: "_StubServer") -> int:
    return int(stub.server.server_address[1])


# ---------------------------------------------------------------------------
# The launcher
# ---------------------------------------------------------------------------


def test_launch_waits_for_ready_then_stops(tmp_path):
    with _StubServer() as stub:
        calls = {}

        def spawn(command, log_handle):
            calls["command"] = command
            return _ready_process()

        # Bind the launcher to the stub's port, so "ready" is the stub's /ready.
        server = launch(["--model", "/ckpt"], host="127.0.0.1", port=_stub_port(stub), ready_timeout=10.0, log_dir=tmp_path, spawn=spawn)
        try:
            assert server.base_url == f"http://127.0.0.1:{_stub_port(stub)}"
            assert server.process.poll() is None
            # The child was handed the linger `serve` command with the resolved port.
            assert calls["command"][:4] == [sys.executable, "-m", "relicllm", "serve"]
            assert "--port" in calls["command"]
        finally:
            server.stop()
        assert server.process.poll() is not None, "stop() must not leave the child running"


def test_launch_reports_a_child_that_exits_before_ready(tmp_path):
    def spawn(command, log_handle):
        log_handle.write(b"boom: checkpoint missing\n")
        log_handle.flush()
        return subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.exit(3)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    with pytest.raises(LaunchError) as excinfo:
        launch(["--model", "/ckpt"], port=0, ready_timeout=10.0, log_dir=tmp_path, spawn=spawn)
    message = str(excinfo.value)
    assert "exited with code 3" in message
    assert "boom: checkpoint missing" in message, "the log tail must be in the error, not just a code"


def test_launch_times_out_when_ready_never_arrives(tmp_path):
    with _StubServer(not_ready_probes=100_000) as stub:
        def spawn(command, log_handle):
            return _ready_process()

        started = time.monotonic()
        with pytest.raises(LaunchError) as excinfo:
            launch(["--model", "/ckpt"], host="127.0.0.1", port=_stub_port(stub), ready_timeout=1.0, log_dir=tmp_path, spawn=spawn)
        assert "not ready within" in str(excinfo.value)
        assert time.monotonic() - started < 10.0


def test_the_ready_check_is_ready_not_health():
    """``/ready`` is 200 only once the backend reports ready; ``/health`` is 200 while merely alive.

    Pinned because polling the wrong one measures a server whose first request pays the model load.
    """
    assert "/ready" in pathlib.Path(launcher_module.__file__).read_text(encoding="utf-8")
    source = pathlib.Path("relicllm/server/openai.py").read_text(encoding="utf-8")
    assert 'if path == "/ready"' in source
    assert 'health.ready' in source


def test_free_port_is_bindable():
    port = launcher_module.free_port()
    assert 0 < port < 65536
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))


# ---------------------------------------------------------------------------
# The parser
# ---------------------------------------------------------------------------


def _bench_serve_parser():
    """The ``bench serve`` parser, where the scenario flag actually lives."""
    from relicllm.cli import build_parser

    bench_parser = next(
        choice for action in build_parser()._subparsers._group_actions
        for choice in (action.choices["bench"],)
    )
    return next(action.choices["serve"] for action in bench_parser._subparsers._group_actions)


def test_scenario_choices_are_the_declared_scenarios():
    parser = _bench_serve_parser()
    action = next(a for a in parser._actions if getattr(a, "dest", None) == "scenario")
    assert set(action.choices) == set(scenarios.names())


def test_client_never_gains_a_launch_flag():
    """The client measures a URL; the launch lives in the bench command, not in the client's parser."""
    parser = _bench_serve_parser()
    for removed in ("--ckpt", "--binary", "--devices", "--sidecar", "--server-drain-seconds"):
        assert removed not in parser._option_string_actions, removed


def test_the_two_ways_to_name_a_server_are_mutually_exclusive(capsys):
    rc = bench_cli.main(["serve", "--base-url", "http://127.0.0.1:9", "--", "--model", "/ckpt"])
    assert rc == bench_cli.EXIT_USAGE
    assert "--base-url" in capsys.readouterr().err


def test_naming_neither_a_model_nor_a_url_is_refused(capsys):
    rc = bench_cli.main(["serve", "--scenario", "decode"])
    assert rc == bench_cli.EXIT_USAGE
    assert "--base-url" in capsys.readouterr().err


def test_the_serve_tail_is_split_at_the_separator():
    head, tail = bench_cli._split_serve_tail(["--scenario", "decode", "--", "--model", "/ckpt", "--backend", "v41"])
    assert head == ["--scenario", "decode"]
    assert tail == ["--model", "/ckpt", "--backend", "v41"]
    # No separator is not the same as an empty tail.
    assert bench_cli._split_serve_tail(["--scenario", "decode"])[1] is None
    assert bench_cli._split_serve_tail(["--"])[1] == []


# ---------------------------------------------------------------------------
# The scenario expansion
# ---------------------------------------------------------------------------


def test_a_scenario_sets_the_client_fields_it_owns_and_leaves_the_rest():
    args = scenarios.default_client_args()
    scenarios.apply(scenarios.resolve("prefill-8k"), args)
    assert args.random_input_len == 8192
    assert args.random_output_len == 32
    assert args.max_concurrency == 1
    # Fields the scenario does not name keep the client default.
    assert args.endpoint == "/v1/completions"
    assert args.token_latch == "content"


def test_scenario_overrides_reject_an_unknown_field():
    with pytest.raises(KeyError):
        scenarios.expand("decode", {"not_a_field": 1})


def test_unknown_scenario_names_the_known_ones():
    with pytest.raises(KeyError) as excinfo:
        scenarios.resolve("nope")
    assert "decode" in str(excinfo.value) and "prefill-8k" in str(excinfo.value)


# ---------------------------------------------------------------------------
# The orchestration, end to end against the stub
# ---------------------------------------------------------------------------


def test_run_launches_measures_records_and_stops(monkeypatch, tmp_path):
    """The whole command with a stub server and an injected spawner: the record names what was run."""
    with _StubServer() as stub:
        processes: list[subprocess.Popen] = []

        def spawn(command, log_handle):
            process = _ready_process()
            processes.append(process)
            return process

        # Pin the launch to the stub's port, so the readiness the launcher waits for is the stub's
        # /ready, and the measurement then hits the stub's streamed completions.
        monkeypatch.setattr(launcher_module, "_spawn_default", spawn)
        out = tmp_path / "bench.json"
        rc = bench_cli.main([
            "serve", "--scenario", "decode", "--json-out", str(out),
            "--endpoint", "/v1/chat/completions",
            "--num-prompts", "2", "--random-input-len", "8", "--random-output-len", "4",
            "--port", str(_stub_port(stub)),
            "--", "--model", "/ckpt", "--backend", "torch",
        ])
        assert rc == bench_cli.EXIT_OK
        record = json.loads(out.read_text())
        # It recorded that it launched, and what.
        assert record["launch"] is not None
        assert "--model" in record["launch"]["argv"]
        assert record["scenarios"]["decode"]["status"] == "pass"
        assert record["tool"] == "relicllm bench"
        # And it tore the child down.
        assert all(p.poll() is not None for p in processes), "the launched server must be stopped"


def test_base_url_mode_records_no_launch(monkeypatch, tmp_path):
    with _StubServer() as stub:
        out = tmp_path / "bench.json"
        # `--base-url` must produce a record whose `launch` is None: nobody knows who started it.
        rc = bench_cli.main([
            "serve", "--base-url", stub.base_url, "--scenario", "decode", "--json-out", str(out),
            "--endpoint", "/v1/chat/completions",
            "--num-prompts", "2", "--random-input-len", "8", "--random-output-len", "4",
        ])
        assert rc == bench_cli.EXIT_OK
        record = json.loads(out.read_text())
        assert record["launch"] is None
        assert record["base_url"] == stub.base_url


# ---------------------------------------------------------------------------
# Dispatch: `relicllm bench` reaches the bench command, not the serve guard
# ---------------------------------------------------------------------------


def test_main_dispatches_bench_without_the_serve_guard(monkeypatch):
    captured = {}

    def fake_run(args, serve_argv=None):
        captured["serve_argv"] = serve_argv
        captured["scenario"] = args.scenario
        return 0

    monkeypatch.setattr(bench_cli, "run", fake_run)
    rc = cli_main(["bench", "serve", "--scenario", "decode", "--", "--model", "/ckpt"])
    assert rc == 0
    assert captured["serve_argv"] == ["--model", "/ckpt"]
    assert captured["scenario"] == ["decode"]


def test_bench_parser_is_registered_on_the_top_level_parser():
    from relicllm.cli import build_parser

    args = build_parser().parse_args(["bench", "serve", "--base-url", "http://127.0.0.1:9"])
    assert args.command == "bench"
    assert args.bench_command == "serve"