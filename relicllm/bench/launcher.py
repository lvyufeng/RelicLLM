"""The launch half of ``relicllm bench``: start ``relicllm serve``, wait until it is ready, stop it.

The measurement client (:mod:`relicllm.bench.client`) talks to a base URL and never starts anything.
This module owns the other decision: which command line started the server. It launches
``relicllm serve`` as a child process, polls ``/ready`` until the backend reports ready, and tears the
process group down afterwards.

**Why a hard-bound port is a required value and not a range.** ``relicllm serve`` takes one integer
``--port``; asking for ephemeral selection means binding a socket here to learn a free number and then
racing the child to bind it. That race is real but bounded (the child binds immediately), and the
alternative -- forcing the operator to pick a free port before every run -- is worse. When
``--port 0`` is given, a free port is picked and reported; the record carries whichever port was used.

**Why ``/ready`` and not ``/health``.** ``relicllm/server/openai.py:268`` serves both; ``/health`` is
200 whenever the process is *alive*, while ``/ready`` is 200 only once the backend reports
``ready`` -- model loaded, ranks joined. A launcher that polled ``/health`` would measure a server
whose first request pays the load. The child dying is caught by ``Popen.poll()`` rather than inferred
from a 503, so a crash is reported as a crash and not as a slow load.
"""

from __future__ import annotations

import os
import pathlib
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

#: Seconds to wait past the child's exit before reading its logs, so the traceback is on disk.
_LOG_DRAIN_SECONDS = 0.5

#: Default readiness timeout. A cold qwen4_exp preload is thousands of seconds a rank on the target
#: box, so the caller is expected to pass one sized to the checkpoint; this is only the floor.
DEFAULT_READY_TIMEOUT = 1800.0


class LaunchError(RuntimeError):
    """The server did not become ready: it exited, or the deadline passed."""


def free_port(host: str = "127.0.0.1") -> int:
    """A currently-free TCP port on ``host``.

    Best-effort: the socket is closed before the caller binds it, so this narrows the window rather
    than closing it. That is the trade named in the module docstring.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return int(probe.getsockname()[1])


def build_serve_command(serve_argv: Sequence[str], *, host: str, port: int) -> list[str]:
    """The argv that starts the server the numbers belong to.

    ``serve_argv`` is everything after the program and ``serve`` -- the runtime flags an operator would
    type. ``--host`` and ``--port`` are placed after them so an explicit value there loses to the
    launcher's, which is the port the client is about to poll. The literal command is stored in the
    record, so "which scheduler do these numbers belong to" is answered by the record.
    """
    return [sys.executable, "-m", "relicllm", "serve", *serve_argv, "--host", host, "--port", str(port)]


@dataclass
class LaunchedServer:
    """A running ``relicllm serve`` child and the log it is writing.

    Used as a context manager: ``with launch(...) as server:`` stops the child on the way out, whether
    the body raised or not.
    """

    process: subprocess.Popen[bytes]
    base_url: str
    command: list[str]
    log_path: pathlib.Path
    shutdown_timeout: float = 30.0
    _log_handle: Any = field(default=None, repr=False)

    def log_tail(self, lines: int = 40) -> str:
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return "(no log)"
        return "\n".join(text.splitlines()[-lines:])

    def stop(self) -> None:
        """SIGTERM the child, then SIGKILL it if it does not leave.

        The child is its own process group (``start_new_session=True``), so a supervised multi-rank
        launch -- where rank 0 spawns its own children -- is signalled as one group rather than leaving
        orphans holding the cards.
        """
        if self.process.poll() is None:
            self._signal_group(signal.SIGTERM)
            try:
                self.process.wait(timeout=self.shutdown_timeout)
            except subprocess.TimeoutExpired:
                self._signal_group(signal.SIGKILL)
                try:
                    self.process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    pass
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def _signal_group(self, sig: int) -> None:
        try:
            os.killpg(os.getpgid(self.process.pid), sig)
        except ProcessLookupError:
            pass

    def __enter__(self) -> "LaunchedServer":
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        self.stop()
        return False


def launch(
    serve_argv: Sequence[str],
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    ready_timeout: float = DEFAULT_READY_TIMEOUT,
    shutdown_timeout: float = 30.0,
    log_dir: str | pathlib.Path | None = None,
    spawn: Callable[..., subprocess.Popen[bytes]] | None = None,
    log_name: str = "relicllm-bench-serve.log",
) -> LaunchedServer:
    """Start ``relicllm serve`` and return it once ``/ready`` answers.

    Raises :class:`LaunchError` with the child's log tail if the child exits before readiness or the
    deadline passes. ``spawn`` is the test seam: the hermetic test substitutes a process that runs a
    stub server instead of a real one.
    """
    if port <= 0:
        port = free_port(host)
    command = build_serve_command(serve_argv, host=host, port=port)
    base_url = f"http://{host}:{port}"

    directory = pathlib.Path(log_dir) if log_dir is not None else pathlib.Path.cwd()
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / log_name
    log_handle = open(log_path, "wb")

    starter = spawn or _spawn_default
    try:
        process = starter(command, log_handle)
    except Exception:
        log_handle.close()
        raise

    server = LaunchedServer(
        process=process,
        base_url=base_url,
        command=command,
        log_path=log_path,
        shutdown_timeout=shutdown_timeout,
        _log_handle=log_handle,
    )
    _wait_for_ready(server, ready_timeout)
    return server


def _spawn_default(command: Sequence[str], log_handle: Any) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        list(command),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )


def _wait_for_ready(server: LaunchedServer, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last = "no response"
    while True:
        code = server.process.poll()
        if code is not None:
            # Let the child's own traceback reach the file before reading it.
            time.sleep(_LOG_DRAIN_SECONDS)
            raise LaunchError(
                f"the server exited with code {code} before it became ready\n"
                f"command: {' '.join(server.command)}\n{server.log_tail()}"
            )
        # Deadline first, probe second: a server that is up but never ready must fail on the clock,
        # not on a probe whose only bound is the socket timeout.
        if time.monotonic() >= deadline:
            raise LaunchError(
                f"the server was not ready within {timeout:.0f}s (last probe: {last})\n"
                f"command: {' '.join(server.command)}\n{server.log_tail()}"
            )
        try:
            with urllib.request.urlopen(server.base_url + "/ready", timeout=2.0) as response:
                if response.status == 200:
                    return
                last = f"HTTP {response.status}"
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code}"
        except (OSError, urllib.error.URLError) as exc:
            last = type(exc).__name__
        time.sleep(0.5)