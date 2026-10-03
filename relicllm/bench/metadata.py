"""What a benchmark record has to carry for the number to be quotable.

``docs/guides/benchmarking.md`` names the metadata a result is required to record -- commit, runtime,
GPU model and count, TP world, host -- because a rate that moves when the host moves is not a number
until it names the host. The client measures; this module supplies the envelope.

Every reader here is **best-effort and never invents a value**: a field that cannot be read is
**absent**, not zero and not ``"unknown"``. ``nvidia-smi`` and ``torch`` are each optional, and on a
box with neither, the record has no ``cuda`` key at all rather than a fabricated device count. That is
the same rule ``relicllm/triage`` states for a missing measurement, and it is why the envelope is
built from a dict that only grows when a probe actually answered.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from typing import Any, Sequence

#: The envelope's shape version. A consumer that reads a record states which one it understands.
SCHEMA = 1


def git_commit(repo: str | None = None) -> str | None:
    """The commit the code under measurement was at, or ``None`` outside a checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def host_metadata() -> dict[str, Any]:
    details: dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }
    return details


def cuda_metadata() -> dict[str, Any]:
    """Device name and count, preferring torch and falling back to ``nvidia-smi``.

    Absent entirely when neither is available -- a record with no GPU should not carry a made-up one.
    """
    found = _cuda_from_torch()
    if found is not None:
        return found
    found = _cuda_from_nvidia_smi()
    return found if found is not None else {}


def _cuda_from_torch() -> dict[str, Any] | None:
    try:
        import torch  # noqa: PLC0415 - optional, and importing it is not free
    except Exception:  # noqa: BLE001 - any import failure means "cannot answer"
        return None
    try:
        if not torch.cuda.is_available():
            return None
        count = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(index) for index in range(count)]
        return {
            "count": count,
            "devices": names,
            "torch": torch.__version__,
            "cuda_runtime": getattr(torch.version, "cuda", None),
        }
    except Exception:  # noqa: BLE001 - a probe that fails answers nothing
        return None


def _cuda_from_nvidia_smi() -> dict[str, Any] | None:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    try:
        output = subprocess.check_output(
            [executable, "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10.0,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    names = [line.strip() for line in output.splitlines() if line.strip()]
    if not names:
        return None
    return {"count": len(names), "devices": names}


def envelope(
    *,
    host: str,
    base_url: str | None,
    launch: dict[str, Any] | None,
    scenarios: dict[str, Any],
    repo: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the record ``relicllm bench`` writes.

    ``launch`` is ``{"argv": [...], "log": "..."}`` when the command started the server, and ``None``
    when ``--base-url`` pointed at one somebody else started -- the two are different claims about the
    number and the envelope keeps them apart rather than filling the gap with a guess.
    """
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "tool": "relicllm bench",
        "host": host_metadata(),
        "base_url": base_url,
        "launch": launch,
        "scenarios": scenarios,
    }
    commit = git_commit(repo)
    if commit is not None:
        record["git_commit"] = commit
    cuda = cuda_metadata()
    if cuda:
        record["cuda"] = cuda
    if extra:
        record.update(extra)
    return record


def dumps(record: dict[str, Any]) -> str:
    return json.dumps(record, indent=2) + "\n"


def describe_command(argv: Sequence[str]) -> str:
    """A copy-pasteable rendering of a launch argv, quoting args that need it."""
    return " ".join(_quote(item) for item in argv)


def _quote(item: str) -> str:
    if item == "" or any(character in item for character in " \t\"'$"):
        return json.dumps(item)
    return item


def env_snapshot(prefix: str = "POCKETLLM_") -> dict[str, str]:
    """The ``POCKETLLM_*`` variables in this process, for the record.

    These are the switches that change where experts live and how the runtime measures itself, so a
    record that does not carry them cannot be reproduced. Kept as a separate call so a caller that
    wants a clean record can omit it.
    """
    return {key: value for key, value in os.environ.items() if key.startswith(prefix)}