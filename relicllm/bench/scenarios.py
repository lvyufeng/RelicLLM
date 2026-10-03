"""Named workloads, so "run the baseline" is one word on a command line.

The roadmap's Phase 0 acceptance is a baseline that covers decode-only and 8k-prefill for every
served runtime. That is two scenarios, and naming them in one place is what keeps two runs
comparable: a scenario is a full set of client arguments, and a caller overriding one field leaves the
rest at the scenario's value rather than at the client's default.

A scenario sets the client flags only. The launch flags (``--model``, ``--backend``,
``--tensor-parallel-size``, …) are the server's and stay the operator's, because two scenarios of one
model are measured against one server.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from relicllm.bench import client


@dataclass(frozen=True)
class Scenario:
    """A workload: prompt and output lengths, request count, arrival rate.

    ``num_prompts`` is small on purpose. The point of a regression baseline is a number that is stable
    run to run, not a saturation number; a hundred prompts at an infinite rate measures the queue, not
    the engine. ``request_rate`` is infinite and ``max_concurrency`` 1 for the same reason -- the
    single-request latency path is the one the roadmap's Phase 2 changes, so it is the one the
    baseline has to track.
    """

    name: str
    description: str
    input_len: int
    output_len: int
    num_prompts: int = 8
    request_rate: float = float("inf")
    max_concurrency: int | None = 1
    warmups: int = 1


#: The two the roadmap names. Decode-only isolates the per-token path; 8k-prefill puts a realistic
#: context on the prefill path at a width the chunked-prefill knob is tuned against.
SCENARIOS: dict[str, Scenario] = {
    "decode": Scenario(
        name="decode",
        description="short prompt, 128 output tokens: the per-token decode path",
        input_len=128,
        output_len=128,
    ),
    "prefill-8k": Scenario(
        name="prefill-8k",
        description="8192-token prompt, 32 output tokens: the long-context prefill path",
        input_len=8192,
        output_len=32,
    ),
}


def names() -> list[str]:
    return sorted(SCENARIOS)


def resolve(name: str) -> Scenario:
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; known: {', '.join(names())}")
    return SCENARIOS[name]


def expand(name: str, overrides: Mapping[str, Any] | None = None) -> Scenario:
    """The scenario, with any named fields replaced."""
    scenario = resolve(name)
    if not overrides:
        return scenario
    unknown = set(overrides) - set(vars(scenario))
    if unknown:
        raise KeyError(f"scenario {name!r} has no field(s) {', '.join(sorted(unknown))}")
    return replace(scenario, **overrides)


def apply(scenario: Scenario, args: Any, *, explicit: set[str] | None = None) -> None:
    """Write a scenario's fields onto a client-args namespace, in place.

    Only the fields the client reads are set; ``base_url`` and ``model`` stay the caller's, so a
    scenario can be run against a launched server exactly as against a running one.

    ``explicit`` names the attributes the operator set on the command line, and those are **not**
    overwritten: the scenario supplies the value for whatever was not named, and an explicit flag wins.
    The caller builds it from the argparse namespace, where an unnamed flag is absent because
    ``add_client_arguments(suppress_defaults=True)`` left it off.
    """
    fields = {
        "random_input_len": scenario.input_len,
        "random_output_len": scenario.output_len,
        "num_prompts": scenario.num_prompts,
        "request_rate": scenario.request_rate,
        "max_concurrency": scenario.max_concurrency,
        "num_warmups": scenario.warmups,
    }
    for name, value in fields.items():
        if explicit and name in explicit:
            continue
        setattr(args, name, value)


def default_client_args() -> Any:
    """A namespace of client defaults, for building a run without a command line."""
    import argparse  # noqa: PLC0415 - only needed on this path

    parser = argparse.ArgumentParser(add_help=False)
    client.add_client_arguments(parser)
    return parser.parse_args([])