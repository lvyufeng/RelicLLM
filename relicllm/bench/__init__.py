"""``relicllm bench`` in two halves, one concern each.

* :mod:`relicllm.bench.client` measures a server over HTTP on vLLM's terms and never starts anything.
* :mod:`relicllm.bench.launcher` starts ``relicllm serve``, waits for readiness and tears it down.

:mod:`relicllm.bench.scenarios` names the workloads the roadmap's baseline is made of, and
:mod:`relicllm.bench.metadata` supplies the envelope a number needs to be quotable. The CLI
(``relicllm bench serve``) is the only place the launch and the measurement are joined, because that is
the only level at which the choice -- launch, or point at a server somebody else started -- is made.
"""

from relicllm.bench.client import RequestOutput, SampleRequest, run_client
from relicllm.bench.launcher import DEFAULT_READY_TIMEOUT, LaunchError, LaunchedServer, launch
from relicllm.bench.metadata import SCHEMA, envelope
from relicllm.bench.scenarios import SCENARIOS, Scenario, expand, resolve

__all__ = [
    "SCENARIOS",
    "SCHEMA",
    "DEFAULT_READY_TIMEOUT",
    "LaunchError",
    "LaunchedServer",
    "RequestOutput",
    "SampleRequest",
    "Scenario",
    "envelope",
    "expand",
    "launch",
    "resolve",
    "run_client",
]