"""The served-path golden fixtures: one real request through one real entry point.

A fixture is a claim that a checkpoint still produces a known answer through a known entry point.
It is the only end-to-end claim this suite makes, and the reason it exists is that everything else
in `tests/` tests a piece: a kernel against a reference, a parser against a fixture, a boundary
against the source tree. A refactor of the request lifecycle can keep every one of those green and
still hand a client different tokens.

What is recorded, and why each field is there, is documented in `tests/README.md`. Two decisions
worth repeating here because they shape the code:

- **The command line is recorded, and the Python entries are driven through the CLI parser.** The
  fixture's `argv` goes through `relicllm.cli.build_parser()` -> `_args()` -> `EngineArgs`, so a
  fixture pins the flag plumbing as well as the model: a flag that stops being read, or a default
  that moves, changes the tokens this test produces. Driving `EngineArgs` directly would leave the
  layer between the operator and the engine untested, which is the layer a serving refactor is
  about.
- **The run is skipped, not failed, when the checkpoint is absent.** A fixture names a path on a
  machine that may not be this one. `pytest.skip` with the missing path is the honest outcome; the
  `test_every_entry_point_has_a_fixture` test in `test_served_path_golden.py` is what keeps the
  *set of recorded fixtures* complete, and it needs no weights at all.

Every entry point is a Python one: the C++ binary's own HTTP front end was removed, and the `cpp`
entry point now goes through `relicllm serve` like the rest. A fixture carries whichever answer its
entry can produce -- token ids exactly, text at least -- and the test compares what is there.

**One entry point is one process**, which is why `run_isolated` spawns a child rather than calling
`run` here. Five engines in one interpreter is a configuration nothing else in this repository uses,
and the failures it produces are not about the fixtures; the docstring on `run_isolated` records the
two that were observed and why they pass alone.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE_DIR = pathlib.Path(__file__).resolve().parent / "fixtures" / "golden"

#: This file, which is also the child launcher: `run_isolated` spawns it with `--entry <name>` and
#: the `__main__` block at the bottom runs exactly one fixture and exits.
GOLDEN_MODULE = pathlib.Path(__file__).resolve()

#: Where a child writes its `Outcome`. A file rather than stdout, because the entry points print
#: freely -- readiness lines, load timings, warnings -- and a result parsed out of that stream would
#: be parsed out of whatever they say next.
CHILD_RESULT_ENV = "POCKETLLM_GOLDEN_RESULT"

#: Where the parent leaves the fixture for the child to run. The child is handed one rather than
#: looking up `fixtures/golden/<entry>.json` itself, so that the fixture a recorder built from
#: `--checkpoint` is the one that runs and not the committed one that happens to share its entry name.
CHILD_FIXTURE_ENV = "POCKETLLM_GOLDEN_FIXTURE"

#: How long a child may run. Generous on purpose: the `v41` fixture pins a 457.8 GiB expert bank and
#: takes forty minutes on a cold segment, and a suite that killed it at ten would report a timeout
#: where it meant to report a fixture.
CHILD_TIMEOUT_SECONDS = 5400.0

#: How much of a failed child's output to keep for the parent's error message.
CHILD_TAIL_LINES = 40

#: Every entry point `tests/README.md` promises a fixture for. The set is asserted by a test, so a
#: new backend cannot be added without recording one.
ENTRY_POINTS = ("v41", "mimo", "xing4", "torch")

#: The opt-in gate for *running* a fixture. The completeness tests need no weights and always run;
#: this one covers the runs, because the cost across the six spans seconds to the better part of an
#: hour and a suite a person runs before a commit cannot include the top of that range by default.
GOLDEN_GATE_ENV = "POCKETLLM_GOLDEN"

DEFAULT_SAMPLING: dict[str, Any] = {"temperature": 0.0, "max_tokens": 16}


def captured_card(
    *,
    platform: str | None = None,
    capability: tuple[int, int] | None = None,
    name: str | None = None,
) -> str:
    """A free-text label for the card that produced an answer, or `""` when it could not be read.

    Recorded, never consumed: nothing compares this string and no skip consults it, so an empty one
    is a gap in provenance and not a host that cannot run. It goes in the fixture beside `commit`
    and `taken_at` for the same reason they do -- so that a future reader looking at a mismatch can
    see what the answer was taken on, which is a question the file otherwise cannot answer.

    The platform is part of the label, and that is the whole point of asking the plane for it first.
    :func:`probe_card_capability` reads ``torch.cuda.get_device_capability``, which on a host with
    no CUDA device answers ``UNKNOWN_CAPABILITY`` -- Turing, the oldest supported cut. That value is
    *right about capability and wrong about identity*: gating off it is safe, recording it as though
    it were the card is a lie about a host nobody measured. Prefixing the platform keeps a rock
    described as a rock, and the three cases below are the honest ones:

    - a CUDA host, where the descriptor can be read: ``cuda 8.9 (Ada / RTX 4090)``;
    - an Ascend host, named as hardware rather than as a compute capability it does not have:
      ``ascend 910B`` -- unreachable today, because no runtime declares ``ascend``, and written for
      the reason :func:`probe_accelerator` already writes a branch nothing can reach;
    - anything else, *including* a CUDA host whose torch build has no device code for the card
      plugged into it: ``""``. The build answers capability ``(7, 5)`` for an sm_89 card it was not
      compiled for, and the descriptor's own ``known`` flag is what separates "read a card I
      recognise" from "read a number I do not believe".
    """
    from relicllm.runtime.device import probe_accelerator, probe_card_capability

    resolved = probe_accelerator().platform if platform is None else platform
    if resolved == "ascend":
        return "ascend 910B"
    if resolved != "cuda":
        return ""

    card = probe_card_capability(capability=capability, name=name)
    if not card.known:
        return ""
    return f"cuda {card.major}.{card.minor} ({card.name})"


# --------------------------------------------------------------------------------------------------
# the recorded form
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldenFixture:
    entry: str
    checkpoint: str
    argv: tuple[str, ...]
    prompt: str
    expected: dict[str, Any]
    env: dict[str, str] = field(default_factory=dict)
    sampling: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_SAMPLING))
    requires: dict[str, int] = field(default_factory=dict)
    commit: str = ""
    taken_at: str = ""
    #: Which card produced the answer, as free text (`cuda 8.9 (Ada / RTX 4090)`), or `""` when it
    #: was not read. **Recorded and never read**: nothing compares it and `unwritable_reason` never
    #: consults it, so its absence is not a reason to skip -- see `captured_card` for why it exists.
    card: str = ""
    notes: str = ""

    @property
    def expected_token_ids(self) -> tuple[int, ...] | None:
        """The recorded ids, `()` when the fixture records an empty answer, `None` when it records none.

        The two are not the same claim and collapsing them is how a fixture stops checking anything:
        `token_ids: []` says "this entry point returns no ids" and is a thing to hold the entry point
        to, while a fixture with no `token_ids` key was recorded before the field and has nothing to
        say. `or ()` would fold the first into the second.
        """
        recorded = self.expected.get("token_ids")
        return None if recorded is None else tuple(recorded)

    @property
    def expected_text(self) -> str | None:
        return self.expected.get("text")

    @property
    def expected_prompt_tokens(self) -> int | None:
        return self.expected.get("prompt_tokens")

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "GoldenFixture":
        missing = [
            key
            for key in ("entry", "checkpoint", "argv", "prompt", "expected")
            if key not in payload
        ]
        if missing:
            raise ValueError(f"golden fixture is missing {', '.join(missing)}")
        return cls(
            entry=str(payload["entry"]),
            checkpoint=str(payload["checkpoint"]),
            argv=tuple(str(item) for item in payload["argv"]),
            prompt=str(payload["prompt"]),
            expected=dict(payload["expected"]),
            env={str(k): str(v) for k, v in (payload.get("env") or {}).items()},
            sampling=dict(payload.get("sampling") or DEFAULT_SAMPLING),
            requires={str(k): int(v) for k, v in (payload.get("requires") or {}).items()},
            commit=str(payload.get("commit") or ""),
            taken_at=str(payload.get("taken_at") or ""),
            card=str(payload.get("card") or ""),
            notes=str(payload.get("notes") or ""),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "entry": self.entry,
            "checkpoint": self.checkpoint,
            "argv": list(self.argv),
            "env": dict(self.env),
            "prompt": self.prompt,
            "sampling": dict(self.sampling),
            "requires": dict(self.requires),
            "expected": dict(self.expected),
            "commit": self.commit,
            "taken_at": self.taken_at,
            "card": self.card,
            "notes": self.notes,
        }

    def unwritable_reason(self) -> str | None:
        """Why this fixture cannot run here, or `None` when it can.

        Two reasons, both about resources and neither about correctness: a checkpoint that is not on
        this machine, and a resource the run needs that this host does not currently have. Everything
        else -- no card, no built extension, a missing engine -- surfaces as the entry point's own
        error, because a fixture whose *answer* moved has to be reported as a mismatch and not as a
        skip.
        """
        path = pathlib.Path(self.checkpoint)
        if not path.exists():
            return f"checkpoint not present: {self.checkpoint}"
        return self._resource_reason()

    def _resource_reason(self) -> str | None:
        """`requires` is the fixture saying what it needs, so the skip is about the host.

        The two `DeepSeek-V4*` and `MiMo*` entry points each pin an expert bank in `/dev/shm`, at 457.8
        and 149.8 GiB, and the two do not fit in a 504 GiB tmpfs at the same time. Without this the
        second one would fail with an out-of-space error somewhere inside a fill loop, which reads as
        a model bug. A fixture that states its requirement turns that into a skip naming the size.
        """
        wanted = self.requires.get("dev_shm_bytes")
        if not wanted:
            return None
        free = _dev_shm_free_bytes()
        if free is None:
            return None
        if free < wanted:
            return (
                f"/dev/shm holds {free / 2**30:.0f} GiB free and this fixture needs "
                f"{wanted / 2**30:.0f} GiB; remove another bank first "
                f"(rm -rf /dev/shm/pocketllm_*_experts*)"
            )
        return None

def _dev_shm_free_bytes() -> int | None:
    """Free space on `/dev/shm`, or `None` when it is not a separate mount.

    A host that runs the container without a tmpfs of its own has no `/dev/shm` to be short of, and
    the segment then lands wherever that host puts shared memory -- so no answer is the right answer.
    """
    try:
        stats = os.statvfs("/dev/shm")
    except OSError:
        return None
    return stats.f_bavail * stats.f_frsize


def fixture_path(entry: str) -> pathlib.Path:
    return FIXTURE_DIR / f"{entry}.json"


def load_fixture(entry: str) -> GoldenFixture | None:
    path = fixture_path(entry)
    if not path.exists():
        return None
    return GoldenFixture.from_json(json.loads(path.read_text(encoding="utf-8")))


def load_fixtures() -> dict[str, GoldenFixture]:
    return {entry: fixture for entry in ENTRY_POINTS if (fixture := load_fixture(entry)) is not None}


def write_fixture(fixture: GoldenFixture) -> pathlib.Path:
    path = fixture_path(fixture.entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fixture.to_json(), indent=2) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------------------
# running one
# --------------------------------------------------------------------------------------------------


@dataclass
class Outcome:
    """What an entry point produced. Either side may be absent; the test compares what is there."""

    token_ids: list[int] | None = None
    text: str | None = None
    prompt_tokens: int | None = None
    elapsed_seconds: float = 0.0
    #: The first decoded step's top-k logits, when the run was asked for them (see
    #: `POCKETLLM_V41_LOGITS_CHECK`). Absent for every entry point that does not record one.
    logits_check: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "token_ids": self.token_ids,
            "text": self.text,
            "prompt_tokens": self.prompt_tokens,
            "elapsed_seconds": self.elapsed_seconds,
            "logits_check": self.logits_check,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "Outcome":
        token_ids = payload.get("token_ids")
        return cls(
            token_ids=None if token_ids is None else [int(item) for item in token_ids],
            text=payload.get("text"),
            prompt_tokens=payload.get("prompt_tokens"),
            elapsed_seconds=float(payload.get("elapsed_seconds") or 0.0),
            logits_check=payload.get("logits_check"),
        )


def run(fixture: GoldenFixture) -> Outcome:
    """Run this fixture **in this process**. The suite does not call this; `run_isolated` does.

    Kept in process because the child has to be able to call it.
    """
    reason = fixture.unwritable_reason()
    if reason is not None:  # pragma: no cover - the test skips before calling this
        raise RuntimeError(reason)
    return _run_python(fixture)


def _with_repo_root_on_the_path() -> str:
    """`PYTHONPATH` with the repository root in front, keeping whatever was there.

    The suite is run from the root and finds `relicllm` because the root is the working directory;
    a child launched by path does not, and neither do the ranks *it* starts.
    """
    existing = os.environ.get("PYTHONPATH")
    return str(REPO_ROOT) if not existing else f"{REPO_ROOT}{os.pathsep}{existing}"


def run_isolated(
    fixture: GoldenFixture,
    *,
    verbose: bool = False,
    extra_env: Mapping[str, str] | None = None,
) -> Outcome:
    """Run one fixture in a child interpreter and return what it produced.

    **One entry point is one process.** Running all six in whatever process pytest happens to be in
    does not work, and the way it fails is not about the fixtures:

    - `xing4` after `mimo` dies inside the CUDA runtime with `resource already mapped`, because the
      previous entry point's host mappings are still registered in this context;
    - `torch` after `mimo` loads its checkpoint with `WORLD_SIZE` read as 1 while the four-way
      sharded model it built expects the shard, and refuses it with a `q8_0 block shape mismatch`.

    Both pass alone, and neither says anything about the checkpoint. A shared CUDA context and a
    shared environment are not what any entry point here is built for -- `docs/guides/benchmarking.md`
    already states the rule as *one process per rank per GPU*, and the operator's own entry point is
    `relicllm serve`, started once per configuration. So the child is the unit of execution and the
    comparison stays in the parent, where a failure can name the fixture.

    The child's output is streamed when `verbose`, and its last lines are kept either way so that a
    child that dies during a load reports what it was doing rather than just a non-zero exit.

    `extra_env` is injected between the fixture's own env and the child-protocol keys. It is how the
    cross-architecture check runs the same fixture under a different fp8/fp4 implementation without
    editing the fixture -- the two legs differ in nothing else.
    """
    reason = fixture.unwritable_reason()
    if reason is not None:
        raise RuntimeError(reason)

    import tempfile

    with tempfile.TemporaryDirectory(prefix="pll-golden-") as tmp:
        # The fixture under test is handed to the child rather than re-read by it. The child used to
        # load `fixtures/golden/<entry>.json` itself, which was correct only while the only caller was
        # the comparison -- there the fixture on disk *is* the fixture under test. The recorder is the
        # other caller, and it runs a fixture it built from `--checkpoint` and has not written yet, so
        # a child that re-read from disk would run the *committed* checkpoint and the recording would
        # describe an answer this launcher never produced. That is a silent mismatch between the file
        # written and the run behind it, which is worse than a failure.
        fixture_path = pathlib.Path(tmp) / "fixture.json"
        fixture_path.write_text(
            json.dumps(fixture.to_json(), indent=2) + "\n", encoding="utf-8"
        )
        result_path = pathlib.Path(tmp) / "outcome.json"
        env = {
            **os.environ,
            **fixture.env,
            **(extra_env or {}),
            CHILD_RESULT_ENV: str(result_path),
            CHILD_FIXTURE_ENV: str(fixture_path),
            # The child is launched by path, so the interpreter would put `tests/` on `sys.path` and
            # never the repository root -- and `relicllm` is imported from the root rather than
            # installed into the environment. PYTHONPATH rather than a `sys.path` insert in the
            # child, because the entry points start their own ranks (`python -c` through
            # `relicllm.supervisor`) and those inherit only what is in the environment.
            "PYTHONPATH": _with_repo_root_on_the_path(),
        }
        process = subprocess.Popen(
            [sys.executable, str(GOLDEN_MODULE), "--entry", fixture.entry],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        killed = threading.Event()
        watchdog = threading.Timer(
            CHILD_TIMEOUT_SECONDS, lambda: (killed.set(), process.kill())
        )
        watchdog.start()
        tail: collections.deque[str] = collections.deque(maxlen=CHILD_TAIL_LINES)
        try:
            assert process.stdout is not None
            for line in process.stdout:
                tail.append(line.rstrip("\n"))
                if verbose:
                    print(line, end="", flush=True)
            code = process.wait()
        finally:
            watchdog.cancel()

        if killed.is_set():
            raise RuntimeError(
                f"the {fixture.entry} fixture was still running after "
                f"{CHILD_TIMEOUT_SECONDS:.0f}s and was killed"
            )
        if code != 0:
            raise RuntimeError(
                f"the {fixture.entry} fixture exited with status {code}; last output:\n"
                + "\n".join(tail)
            )
        if not result_path.exists():
            raise RuntimeError(
                f"the {fixture.entry} fixture exited 0 without writing a result to "
                f"{CHILD_RESULT_ENV}"
            )
        return Outcome.from_json(json.loads(result_path.read_text(encoding="utf-8")))


def _sampling_params(fixture: GoldenFixture):
    from relicllm.api import SamplingParams

    allowed = {"max_tokens", "temperature", "top_p", "top_k", "min_p", "seed", "stop"}
    unknown = sorted(set(fixture.sampling) - allowed)
    if unknown:
        raise ValueError(f"fixture sampling has unsupported keys: {', '.join(unknown)}")
    return SamplingParams(**fixture.sampling)


def _engine_args_from_argv(fixture: GoldenFixture):
    """Parse the fixture's command line the way the CLI does.

    `build_parser`/`_args` is the same pair `relicllm` itself calls, so a fixture exercises the
    operator's path into the engine rather than a constructor nobody types.
    """
    from relicllm.cli import _args, build_parser

    argv = list(fixture.argv)
    if not argv or argv[0] != "serve":
        argv = ["serve", *argv]
    namespace = build_parser().parse_args(argv)
    return _args(namespace)


def _run_python(fixture: GoldenFixture) -> Outcome:
    from relicllm import LLM

    env = dict(fixture.env)
    previous = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    started = time.perf_counter()
    llm = None
    try:
        llm = LLM(_engine_args_from_argv(fixture))
        result = llm.generate([fixture.prompt], _sampling_params(fixture))[0]
    finally:
        if llm is not None:
            llm.close()
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    logits_check = result.metadata.get("logits_check")
    prompt_tokens = int(getattr(result.usage, "prompt_tokens", 0)) or None
    if logits_check is not None:
        # The spec's record names the prompt token count so a mismatch in what was compared is
        # visible; the backend cannot know it, the run can.
        logits_check = {**logits_check, "prompt_tokens": prompt_tokens or 0}
    return Outcome(
        token_ids=list(result.token_ids),
        text=result.text,
        prompt_tokens=prompt_tokens,
        elapsed_seconds=time.perf_counter() - started,
        logits_check=logits_check,
    )


# --------------------------------------------------------------------------------------------------
# the child side
# --------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run one fixture and write its outcome where the parent asked for it.

    Not a user-facing command -- `run_isolated` is the only caller, and it is what makes one entry
    point one process. The outcome goes to `CHILD_RESULT_ENV` rather than to stdout, and the env var
    is required rather than defaulted, so a child launched by hand says so instead of printing a JSON
    blob into a stream the entry point is also writing to.
    """
    parser = argparse.ArgumentParser(description="Run one served-path golden fixture")
    parser.add_argument("--entry", required=True, choices=list(ENTRY_POINTS))
    args = parser.parse_args(argv)

    # The fixture comes from the parent, not from `fixtures/golden/<entry>.json`. The recorder runs a
    # fixture it has not written yet, so reading the file by name would run the *committed* checkpoint
    # while the recording described this one -- a mismatch between the answer recorded and the run
    # behind it, and silent, because both are the same entry point. The path is required rather than
    # defaulted for the same reason the result path is: a child run by hand should say it was not
    # handed a fixture, and the state that means "use whatever is on disk" is one `run_isolated`
    # never wants.
    fixture_env = os.environ.get(CHILD_FIXTURE_ENV)
    if not fixture_env:
        raise SystemExit(
            f"{CHILD_FIXTURE_ENV} is not set, so there is no fixture to run. This module is spawned "
            f"by tests/golden_fixtures.py::run_isolated, which writes the fixture and sets it"
        )
    fixture_path = pathlib.Path(fixture_env)
    if not fixture_path.exists():
        raise SystemExit(f"the fixture named by {CHILD_FIXTURE_ENV} is not there: {fixture_env}")
    fixture = GoldenFixture.from_json(json.loads(fixture_path.read_text(encoding="utf-8")))
    if fixture.entry != args.entry:
        raise SystemExit(
            f"the fixture at {fixture_env} is for {fixture.entry!r}, not {args.entry!r}"
        )

    destination = os.environ.get(CHILD_RESULT_ENV)
    if not destination:
        raise SystemExit(
            f"{CHILD_RESULT_ENV} is not set, so there is nowhere to write the outcome. This module "
            f"is spawned by tests/golden_fixtures.py::run_isolated, which sets it"
        )

    outcome = run(fixture)
    pathlib.Path(destination).write_text(json.dumps(outcome.to_json()), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
