"""The measurement half of ``relicllm bench``: a client for an OpenAI-compatible server.

This is the vLLM-convention serving benchmark, on the terms defined once in
``docs/guides/latency_metrics.md``: TTFT, TPOT, ITL, E2EL, throughput and goodput, with the
formulas, latching rules, percentile set and goodput semantics ``vllm bench serve`` uses, so a
RelicLLM row and a vLLM row can go in one table without an argument about what the words mean.

The quantity is **what a client observes**, measured over HTTP against a base URL. It knows nothing
about which runtime produced it and it never starts anything -- the launch is
:mod:`relicllm.bench.launcher`'s job and the CLI owns the orchestration, so **this module measures a
server somebody else started** and a caller that has one already passes ``--base-url``.

Two deliberate departures from a literal copy of vLLM's client, both reported rather than hidden:

* **The first SSE chunk is not the first token.** vLLM's client latches TTFT on the first chunk with a
  non-empty ``choices`` array, tolerating empty text (``endpoint_request_func.py:236-245``). A RelicLLM
  server writes a role-only delta before the backend is even called, so that chunk arrives before any
  prefill work has happened and crediting it understates TTFT by the entire queue-and-prefill cost.
  This client measures both and reports both; which one is the headline is ``--token-latch``
  (default ``content``, the honest one).
* **Streaming ``output_tokens``.** vLLM takes the count from a usage chunk and otherwise re-tokenizes
  the generated text. RelicLLM's servers emit no usage chunk while streaming, so the fallback applies;
  pass ``--tokenizer`` for the exact vLLM behaviour, otherwise the token-bearing chunk count is used
  and the JSON record says which source was used.

No third-party HTTP client: ``prometheus_client``, ``aiohttp`` and ``vllm`` are not installed in
either development environment. ``numpy`` is used for the statistics because vLLM's are ``numpy``'s
(population standard deviation, linear-interpolated percentiles) and matching them by hand would be a
bug farm.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import pathlib
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import numpy as np

MILLISECONDS_TO_SECONDS = 1000.0

#: The port ``relicllm serve`` binds by default, and the one the other server benchmarks point at.
#: A convenience default rather than the only way to name a server: ``--base-url`` overrides it.
DEFAULT_BASE_URL = "http://127.0.0.1:8123"

# The metrics ``--goodput`` accepts, in the order vLLM zips them (``vllm/benchmarks/serve.py:638-660``).
GOODPUT_KEYS = ("ttft", "tpot", "e2el")

# ``--percentile-metrics`` defaults to these for a generative endpoint
# (``vllm/benchmarks/serve.py:1819-1834``).
DEFAULT_PERCENTILE_METRICS = "ttft,tpot,itl"


# ---------------------------------------------------------------------------
# The shared client primitives (also re-exported by tests/serve_client.py, so
# the prompt text and the definition of a request are one implementation).
# ---------------------------------------------------------------------------


@dataclass
class HttpResult:
    status: int
    body: bytes
    elapsed_seconds: float
    first_event_seconds: float | None = None

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")

    def json(self) -> dict[str, Any]:
        value = json.loads(self.text)
        if not isinstance(value, dict):
            raise AssertionError(f"expected JSON object, got {type(value).__name__}")
        return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def http_request(
    base_url: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float,
) -> HttpResult:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers=headers,
        method="POST" if payload is not None else "GET",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            return HttpResult(response.status, body, time.perf_counter() - started)
    except urllib.error.HTTPError as exc:
        return HttpResult(exc.code, exc.read(), time.perf_counter() - started)


def prompt_text(index: int, words: int) -> str:
    sentence = (
        "Explain the following benchmark request in one concise paragraph. "
        "The quick brown fox jumps over the lazy dog while the deployment team "
        "checks scheduler fairness, KV cache isolation, and tensor parallel safety. "
    )
    text = (f"Request {index}: " + sentence) * max(1, words // 35 + 1)
    return text[: max(64, words * 5)]


# ---------------------------------------------------------------------------
# One request's measurement
# ---------------------------------------------------------------------------


@dataclass
class RequestOutput:
    """One request's client-side measurement.

    Field names and meaning follow ``RequestFuncOutput``
    (``vllm/benchmarks/lib/endpoint_request_func.py:88-100``); the ``*_first_chunk`` fields are the
    RelicLLM addition described in the module docstring.
    """

    success: bool = False
    error: str = ""
    start_time: float = 0.0
    ttft: float = 0.0
    ttft_first_chunk: float = 0.0
    itl: list[float] = field(default_factory=list)
    latency: float = 0.0
    stream_end: float = 0.0
    output_tokens: int = 0
    prompt_tokens: int | None = None
    generated_text: str = ""
    nominal_input_len: int = 0
    nominal_output_len: int = 0
    request_id: int = 0
    token_events: int = 0

    def row(self, origin: float = 0.0) -> dict[str, Any]:
        """This request's measurements, with times relative to ``origin``.

        ``start_seconds`` is the client's own send time, which is the one figure ``ttft_seconds`` and
        ``itl_seconds`` cannot stand in for: those are relative to the request, so a run that issues
        more prompts than it holds in flight -- a refilled batch -- has no way to place a request on a
        shared axis without it. ``origin`` is the earliest send of the run, so the field is an offset
        from the first request, not a clock reading.
        """
        return {
            "request_id": self.request_id,
            "success": self.success,
            "error": self.error,
            "start_seconds": self.start_time - origin,
            "ttft_seconds": self.ttft,
            "ttft_first_chunk_seconds": self.ttft_first_chunk,
            "itl_seconds": list(self.itl),
            "latency_seconds": self.latency,
            "stream_end_seconds": self.stream_end,
            "output_tokens": self.output_tokens,
            "prompt_tokens": self.prompt_tokens,
            "token_events": self.token_events,
            "generated_chars": len(self.generated_text),
        }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class SampleRequest:
    prompt: str
    expected_output_len: int
    nominal_input_len: int


def build_dataset(args: argparse.Namespace, rng: np.random.Generator) -> list[SampleRequest]:
    """``--dataset-name random`` in vLLM's shape, or the repository's own prompts.

    vLLM samples real token ids from the tokenizer's vocabulary
    (``vllm/benchmarks/datasets/datasets.py:585-640``) and decodes them to text, so its input length is
    exact. Without a tokenizer this client cannot be exact, so ``random`` reproduces the *shape*: a
    uniform draw from ``len * (1 - ratio) .. len * (1 + ratio)``, converted to words at
    ``--chars-per-token``. The record therefore carries ``nominal_input_len``, and the real counts come
    from the server's ``usage`` when it reports them.
    """
    if args.dataset_name == "custom":
        return [
            SampleRequest(prompt_text(i, args.random_input_len), args.random_output_len, args.random_input_len)
            for i in range(args.num_prompts)
        ]

    inputs = rng.uniform(1.0 - args.random_range_ratio, 1.0 + args.random_range_ratio, args.num_prompts)
    outputs = rng.uniform(1.0 - args.random_range_ratio, 1.0 + args.random_range_ratio, args.num_prompts)
    requests = []
    for index in range(args.num_prompts):
        input_len = max(1, int(round(args.random_input_len * float(inputs[index]))))
        output_len = max(1, int(round(args.random_output_len * float(outputs[index]))))
        words = max(1, int(round(input_len * args.chars_per_token / 6.0)))
        requests.append(SampleRequest(prompt_text(index, words), output_len, input_len))
    return requests


def build_payload(
    args: argparse.Namespace,
    model: str,
    sample: SampleRequest,
    *,
    stream: bool,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    output_len = sample.expected_output_len if max_tokens is None else max_tokens
    base: dict[str, Any] = {
        "model": model,
        "max_tokens": output_len,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": stream,
    }
    if args.endpoint.endswith("/chat/completions"):
        base["messages"] = [{"role": "user", "content": sample.prompt}]
    else:
        base["prompt"] = sample.prompt
    if args.extra_body:
        base.update(json.loads(args.extra_body))
    return base


# ---------------------------------------------------------------------------
# Request functions
# ---------------------------------------------------------------------------


def _chunk_text(data: dict[str, Any], endpoint: str) -> str:
    """The text a chunk carries, or "" if it carries no token.

    ``delta.reasoning_content`` is included because RelicLLM serves a reasoning channel in the same
    chunk stream; a token is a token.
    """
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0]
    if not isinstance(choice, dict):
        return ""
    if endpoint.endswith("/chat/completions"):
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            return ""
        content = delta.get("content")
        reasoning = delta.get("reasoning_content")
        if isinstance(content, str) and content:
            return content
        if isinstance(reasoning, str) and reasoning:
            return reasoning
        return ""
    text = choice.get("text")
    return text if isinstance(text, str) else ""


def stream_request(base_url: str, endpoint: str, payload: dict[str, Any], *, timeout: float, latch: str) -> RequestOutput:
    """One streaming request, timestamped at token arrival.

    The latching rules are ``endpoint_request_func.py:226-262`` for completions and ``:403-436`` for
    chat, with ``latch`` choosing what counts as a token event:

    * ``first-chunk`` - any chunk with a non-empty ``choices`` array, which is upstream's rule exactly,
      empty text included.
    * ``content`` - a chunk whose text is non-empty. TTFT then means "the first token the client could
      read", which is the number a user experiences.
    """
    out = RequestOutput()
    request = urllib.request.Request(
        base_url.rstrip("/") + endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    out.start_time = started
    most_recent_timestamp = started
    latched = False
    first_chunk_seen = False
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            while True:
                line = response.readline()
                if not line:
                    break
                if not line.startswith(b"data: "):
                    continue
                raw = line[6:].strip()
                if raw == b"[DONE]":
                    out.stream_end = time.perf_counter() - started
                    break
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(data, dict):
                    continue
                if not data.get("choices"):
                    usage = data.get("usage")
                    if isinstance(usage, dict):
                        out.output_tokens = int(usage.get("completion_tokens") or 0)
                        prompt_tokens = usage.get("prompt_tokens")
                        if prompt_tokens is not None:
                            out.prompt_tokens = int(prompt_tokens)
                    continue

                timestamp = time.perf_counter()
                if not first_chunk_seen:
                    first_chunk_seen = True
                    out.ttft_first_chunk = timestamp - started

                text = _chunk_text(data, endpoint)
                out.generated_text += text
                is_event = True if latch == "first-chunk" else bool(text)
                if is_event:
                    if not latched:
                        latched = True
                        out.ttft = timestamp - started
                    else:
                        out.itl.append(timestamp - most_recent_timestamp)
                    most_recent_timestamp = timestamp
                    out.token_events += 1
                    if latch == "content":
                        out.output_tokens += 1
            if out.stream_end == 0.0:
                out.stream_end = time.perf_counter() - started
    except urllib.error.HTTPError as exc:
        out.error = f"HTTP {exc.code}: {exc.read()[:200]!r}"
        out.latency = time.perf_counter() - started
        return out
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        out.error = f"{type(exc).__name__}: {exc}"
        out.latency = time.perf_counter() - started
        return out

    if not latched:
        out.error = "Never received a valid chunk to calculate TTFT."
        out.latency = most_recent_timestamp - started
        return out
    out.success = True
    out.latency = most_recent_timestamp - started
    return out


def nonstream_request(base_url: str, endpoint: str, payload: dict[str, Any], *, timeout: float) -> RequestOutput:
    """One blocking request. No per-token boundary exists, so no ITL is invented.

    TTFT and E2EL are the whole round trip here, which is what vLLM records for a non-streaming
    endpoint too (``endpoint_request_func.py:625``).
    """
    out = RequestOutput()
    started = time.perf_counter()
    out.start_time = started
    try:
        result = http_request(base_url, endpoint, payload, timeout=timeout)
        elapsed = time.perf_counter() - started
        out.latency = elapsed
        out.stream_end = elapsed
        out.ttft = elapsed
        out.ttft_first_chunk = elapsed
        if result.status != 200:
            out.error = f"HTTP {result.status}: {result.text[:200]}"
            return out
        body = result.json()
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        out.error = f"{type(exc).__name__}: {exc}"
        out.latency = time.perf_counter() - started
        return out

    choices = body.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        choice = choices[0]
        message = choice.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            out.generated_text = message["content"]
        elif isinstance(choice.get("text"), str):
            out.generated_text = choice["text"]
    usage = body.get("usage")
    if isinstance(usage, dict):
        out.output_tokens = int(usage.get("completion_tokens") or 0)
        if usage.get("prompt_tokens") is not None:
            out.prompt_tokens = int(usage["prompt_tokens"])
    out.success = True
    return out


def run_one(
    base_url: str,
    endpoint: str,
    payload: dict[str, Any],
    *,
    timeout: float,
    stream: bool,
    latch: str,
) -> RequestOutput:
    if stream:
        return stream_request(base_url, endpoint, payload, timeout=timeout, latch=latch)
    return nonstream_request(base_url, endpoint, payload, timeout=timeout)


# ---------------------------------------------------------------------------
# Arrival schedule
# ---------------------------------------------------------------------------


def arrival_delays(num_requests: int, request_rate: float, burstiness: float, rng: np.random.Generator) -> list[float]:
    """Request start offsets in seconds.

    Mirrors ``get_request`` (``vllm/benchmarks/serve.py:438-490``): intervals are
    ``gamma(shape=burstiness, scale=1/(rate*burstiness))``, so burstiness 1 is a Poisson process and
    infinity tends to a constant interval; the cumulative series is then rescaled so the last arrival
    lands on ``num/rate``, which is what keeps throughput stable across seeds.
    """
    require(burstiness > 0, f"burstiness must be positive, got {burstiness}")
    if request_rate == float("inf"):
        return [0.0] * num_requests

    delays: list[float] = []
    if burstiness == float("inf"):
        delays = [1.0 / request_rate] * num_requests
    else:
        theta = 1.0 / (request_rate * burstiness)
        delays = [float(rng.gamma(burstiness, theta)) for _ in range(num_requests)]
    for i in range(1, len(delays)):
        delays[i] += delays[i - 1]
    if delays and delays[-1] != 0.0:
        factor = (num_requests / request_rate) / delays[-1]
        delays = [delay * factor for delay in delays]
    return delays


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _summarise(values: list[float], percentiles: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "percentiles": {}}
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "percentiles": {p: float(np.percentile(array, p)) for p in percentiles},
    }


def calculate_metrics(
    outputs: list[RequestOutput],
    dur_s: float,
    *,
    percentiles: list[float],
    goodput_config: dict[str, float] | None,
) -> dict[str, Any]:
    """The vLLM summary, computed in vLLM's order and with its exclusions.

    ``vllm/benchmarks/serve.py:588-782``: TPOT is ``(latency - ttft) / (n - 1)`` and is excluded from
    the TPOT summary when ``n <= 1`` (it is still fed to goodput as 0.0); ITL pools every gap of every
    request; E2EL is ``latency``; goodput counts a request only when it meets *every* configured SLO.
    """
    itls: list[float] = []
    tpots: list[float] = []
    all_tpots: list[float] = []
    ttfts: list[float] = []
    ttfts_first_chunk: list[float] = []
    e2els: list[float] = []
    stream_ends: list[float] = []
    output_lens: list[int] = []
    total_input = 0
    input_known = True
    completed = 0
    good_completed = 0

    for output in outputs:
        if not output.success:
            output_lens.append(0)
            continue
        output_len = output.output_tokens
        if output_len <= 0:
            # vLLM falls back to re-tokenizing the generated text; with no tokenizer and a
            # non-streaming response that yields nothing, keep the request out of the token-based
            # figures rather than invent one.
            output_len = max(output.token_events, 1)
        output_lens.append(output_len)
        tpot = 0.0
        if output_len > 1:
            tpot = (output.latency - output.ttft) / (output_len - 1)
            tpots.append(tpot)
        all_tpots.append(tpot)
        itls += output.itl
        ttfts.append(output.ttft)
        ttfts_first_chunk.append(output.ttft_first_chunk)
        e2els.append(output.latency)
        stream_ends.append(output.stream_end)
        if output.prompt_tokens is None:
            input_known = False
        else:
            total_input += output.prompt_tokens
        completed += 1

    if goodput_config:
        missing = [key for key in goodput_config if key not in GOODPUT_KEYS]
        require(not missing, f"unknown goodput metric(s): {missing}; allowed: {list(GOODPUT_KEYS)}")
        series = {"ttft": ttfts, "tpot": all_tpots, "e2el": e2els}
        keys = [key for key in GOODPUT_KEYS if key in goodput_config]
        slo = {key: goodput_config[key] / MILLISECONDS_TO_SECONDS for key in keys}
        for index in range(completed):
            if all(series[key][index] <= slo[key] for key in keys):
                good_completed += 1

    max_output_tokens_per_s = 0.0
    max_concurrent_requests = 0
    successful = [output for output in outputs if output.success]
    if successful:
        min_start = min(output.start_time for output in successful)
        max_end = max(output.start_time + output.latency for output in successful)
        bucket_count = int(math.ceil(max_end - min_start)) + 1
        tokens_per_second = np.zeros(bucket_count)
        concurrent_per_second = np.zeros(bucket_count)
        for output in successful:
            current = output.start_time + output.ttft
            token_times = [current]
            for gap in output.itl:
                current += gap
                token_times.append(current)
            for token_time in token_times:
                bucket = int(token_time - min_start)
                if 0 <= bucket < bucket_count:
                    tokens_per_second[bucket] += 1
            first_bucket = int(output.start_time - min_start)
            last_bucket = int((output.start_time + output.latency) - min_start)
            for bucket in range(first_bucket, last_bucket + 1):
                concurrent_per_second[bucket] += 1
        max_output_tokens_per_s = float(np.max(tokens_per_second))
        max_concurrent_requests = int(np.max(concurrent_per_second))

    total_output = sum(output_lens)
    metrics: dict[str, Any] = {
        "completed": completed,
        "failed": len(outputs) - completed,
        "duration_seconds": dur_s,
        "total_input_tokens": total_input if input_known else None,
        "total_output_tokens": total_output,
        "request_throughput": completed / dur_s if dur_s > 0 else 0.0,
        "request_goodput": (good_completed / dur_s if dur_s > 0 else 0.0) if goodput_config else None,
        "output_throughput": total_output / dur_s if dur_s > 0 else 0.0,
        "total_token_throughput": ((total_input + total_output) / dur_s if input_known and dur_s > 0 else None),
        "max_output_tokens_per_s": max_output_tokens_per_s,
        "max_concurrent_requests": max_concurrent_requests,
        "ttft": _summarise(ttfts, percentiles),
        "ttft_first_chunk": _summarise(ttfts_first_chunk, percentiles),
        "tpot": _summarise(tpots, percentiles),
        "itl": _summarise(itls, percentiles),
        "e2el": _summarise(e2els, percentiles),
        "stream_end": _summarise(stream_ends, percentiles),
        "output_lens": output_lens,
    }
    return metrics


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def warm_up(
    base_url: str,
    endpoint: str,
    model: str,
    sample: SampleRequest,
    *,
    count: int,
    args: argparse.Namespace,
) -> int:
    """Discarded requests before the measured run.

    The engine's first request is not free -- kernel module load, allocator growth, and on multi-rank
    runs the first collective -- and it is not the thing being measured. ``--num-warmups`` defaults to
    0, matching vLLM; the concurrency harness defaults to 1 round for the same reason.
    """
    sent = 0
    for _ in range(count):
        payload = build_payload(args, model, sample, stream=args.stream)
        outcome = run_one(base_url, endpoint, payload, timeout=args.timeout, stream=args.stream, latch=args.token_latch)
        if outcome.success:
            sent += 1
    return sent


def run_measured(
    base_url: str,
    endpoint: str,
    model: str,
    requests: list[SampleRequest],
    args: argparse.Namespace,
) -> tuple[list[RequestOutput], float]:
    rng = np.random.default_rng(args.seed)
    delays = arrival_delays(len(requests), args.request_rate, args.burstiness, rng)
    max_workers = args.max_concurrency or len(requests)
    max_workers = max(1, min(max_workers, args.max_workers_cap))
    outputs: dict[int, RequestOutput] = {}
    futures: list[concurrent.futures.Future[None]] = []

    def one(index: int) -> None:
        payload = build_payload(args, model, requests[index], stream=args.stream)
        output = run_one(base_url, endpoint, payload, timeout=args.timeout, stream=args.stream, latch=args.token_latch)
        output.request_id = index
        output.nominal_input_len = requests[index].nominal_input_len
        output.nominal_output_len = requests[index].expected_output_len
        outputs[index] = output

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        for index, delay in enumerate(delays):
            if delay > 0:
                wait = started + delay - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
            futures.append(pool.submit(one, index))
        concurrent.futures.wait(futures)
    duration = time.perf_counter() - started
    # A worker that raised before it could build a RequestOutput would otherwise vanish from the result
    # set and be reported as a request that never happened. Fail the run instead: the harness exists to
    # produce numbers, and a silently short request list is a wrong number.
    for future in futures:
        error = future.exception()
        if error is not None:
            raise error
    return [outputs[index] for index in sorted(outputs)], duration


def print_summary(args: argparse.Namespace, metrics: dict[str, Any]) -> None:
    label = "{:<40} {:<10}"
    print(f"{'Successful requests:':<40} {metrics['completed']:<10}")
    print(f"{'Failed requests:':<40} {metrics['failed']:<10}")
    if args.max_concurrency is not None:
        print(label.format("Maximum request concurrency:", args.max_concurrency))
    if args.request_rate != float("inf"):
        print(label.format("Request rate configured (RPS):", f"{args.request_rate:.2f}"))
    print(label.format("Benchmark duration (s):", f"{metrics['duration_seconds']:.2f}"))
    print(label.format("Total input tokens:", metrics["total_input_tokens"] if metrics["total_input_tokens"] is not None else "n/a"))
    print(label.format("Total generated tokens:", metrics["total_output_tokens"]))
    print(label.format("Request throughput (req/s):", f"{metrics['request_throughput']:.2f}"))
    if metrics["request_goodput"] is not None:
        print(label.format("Request goodput (req/s):", f"{metrics['request_goodput']:.2f}"))
    print(label.format("Output token throughput (tok/s):", f"{metrics['output_throughput']:.2f}"))
    if metrics["total_token_throughput"] is not None:
        print(label.format("Total token throughput (tok/s):", f"{metrics['total_token_throughput']:.2f}"))
    print(label.format("Peak output token throughput (tok/s):", f"{metrics['max_output_tokens_per_s']:.2f}"))
    print(label.format("Peak concurrent requests:", metrics["max_concurrent_requests"]))

    selected = selected_percentile_metrics(args)
    for key, name, header in (
        ("ttft", "TTFT", "Time to First Token"),
        ("tpot", "TPOT", "Time per Output Token (excl. 1st token)"),
        ("itl", "ITL", "Inter-token Latency"),
        ("e2el", "E2EL", "End-to-end Latency"),
    ):
        if key not in selected:
            continue
        summary = metrics[key]
        print("{s:-^50}".format(s=header))
        print(label.format(f"Mean {name} (ms):", f"{summary['mean'] * 1000:.2f}"))
        print(label.format(f"Median {name} (ms):", f"{summary['median'] * 1000:.2f}"))
        for percentile, value in sorted(summary["percentiles"].items()):
            word = str(int(percentile)) if float(percentile).is_integer() else str(percentile)
            print(label.format(f"P{word} {name} (ms):", f"{value * 1000:.2f}"))
    if args.token_latch == "first-chunk":
        return
    # The honest TTFT and vLLM's TTFT are different numbers on RelicLLM's servers; printing the gap
    # makes that visible instead of arguable.
    first_chunk = metrics["ttft_first_chunk"]
    print("{s:-^50}".format(s="Role-chunk cost (first chunk -> first token)"))
    print(label.format("Mean TTFT first chunk (ms):", f"{first_chunk['mean'] * 1000:.2f}"))
    print(label.format("Mean difference (ms):", f"{(metrics['ttft']['mean'] - first_chunk['mean']) * 1000:.2f}"))


def selected_percentile_metrics(args: argparse.Namespace) -> set[str]:
    raw = args.percentile_metrics or DEFAULT_PERCENTILE_METRICS
    return {item.strip() for item in raw.split(",") if item.strip()}


def parse_percentiles(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_goodput(values: list[str] | None) -> dict[str, float] | None:
    if not values:
        return None
    config: dict[str, float] = {}
    for item in values:
        require(":" in item, f'--goodput entries are "KEY:VALUE" pairs in milliseconds, got {item!r}')
        key, _, raw = item.partition(":")
        config[key.strip()] = float(raw)
    return config


def discover_model(base_url: str) -> str:
    result = http_request(base_url, "/v1/models", timeout=30.0)
    require(result.status == 200, f"GET /v1/models returned HTTP {result.status}: {result.text[:200]}")
    data = result.json().get("data")
    require(isinstance(data, list) and data, "/v1/models returned no models")
    return str(data[0]["id"])


def apply_tokenizer_counts(outputs: list[RequestOutput], tokenizer_path: str) -> int:
    """Re-tokenize generated text, as vLLM does when no usage chunk arrives.

    ``vllm/benchmarks/serve.py:608-622``. vLLM's own note is worth repeating: this can inflate the
    count slightly, because a server may hold a partial token back and emit it with the next chunk.
    Returns how many requests it touched.
    """
    from transformers import AutoTokenizer  # noqa: PLC0415 - optional dependency

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    touched = 0
    for output in outputs:
        if not output.success or not output.generated_text:
            continue
        output.output_tokens = len(tokenizer(output.generated_text, add_special_tokens=False).input_ids)
        touched += 1
    return touched


def run_client(args: argparse.Namespace) -> dict[str, Any]:
    """Measure one server at one workload and return the vLLM-shaped record.

    The record is exactly what ``tests/bench_serving.py`` produced, minus ``script`` (the caller owns
    naming) and with ``git_commit`` carried by the caller's envelope instead. ``--base-url`` may be
    omitted on the namespace when the caller has already resolved it (the CLI resolves an ephemeral
    port after launching); the caller then sets ``base_url`` on the namespace.
    """
    require(args.num_prompts > 0, "--num-prompts must be positive")
    metrics_percentiles = parse_percentiles(args.metric_percentiles)
    goodput_config = parse_goodput(args.goodput)

    started = time.perf_counter()
    record: dict[str, Any] = {
        "endpoint": args.endpoint,
        "stream": args.stream,
        "token_latch": args.token_latch,
        "dataset_name": args.dataset_name,
        "num_prompts": args.num_prompts,
        "request_rate": args.request_rate,
        "burstiness": args.burstiness,
        "max_concurrency": args.max_concurrency,
        "num_warmups": args.num_warmups,
        "random_input_len": args.random_input_len,
        "random_output_len": args.random_output_len,
        "random_range_ratio": args.random_range_ratio,
        "seed": args.seed,
        "goodput_slo_ms": goodput_config,
        "metric_percentiles": metrics_percentiles,
        "percentile_metrics": sorted(selected_percentile_metrics(args)),
    }
    try:
        base_url = args.base_url.rstrip("/")
        record["base_url"] = base_url
        model = args.model or discover_model(base_url)
        record["model"] = model

        if args.num_warmups > 0:
            rng = np.random.default_rng(args.seed)
            prompts = build_dataset(args, rng)
            record["warmup_successes"] = warm_up(
                base_url, args.endpoint, model, prompts[0], count=args.num_warmups, args=args
            )

        rng = np.random.default_rng(args.seed)
        requests = build_dataset(args, rng)
        outputs, duration = run_measured(base_url, args.endpoint, model, requests, args)
        if args.tokenizer:
            record["tokenized_requests"] = apply_tokenizer_counts(outputs, args.tokenizer)
            record["output_tokens_source"] = "tokenizer"
        else:
            record["output_tokens_source"] = "token-bearing chunks"
        metrics = calculate_metrics(outputs, duration, percentiles=metrics_percentiles, goodput_config=goodput_config)
        metrics["input_tokens_source"] = "server usage" if all(o.prompt_tokens is not None for o in outputs) else "unavailable"
        record["metrics"] = metrics
        # The rows are offset from the earliest send, which is the same origin `calculate_metrics` used
        # for its per-second concurrency series.
        origin = min((o.start_time for o in outputs if o.success), default=0.0)
        record["requests"] = [output.row(origin) for output in outputs]
        record["status"] = "pass" if metrics["failed"] == 0 else "partial"
    except Exception as exc:
        record.update({"status": "fail", "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        record["elapsed_seconds"] = time.perf_counter() - started

    print_summary(args, record["metrics"])
    return record


# ---------------------------------------------------------------------------
# The client's argument surface, named as vLLM names it.
# ---------------------------------------------------------------------------


def add_client_arguments(parser: argparse.ArgumentParser, *, suppress_defaults: bool = False) -> None:
    """Declare the client half's flags on ``parser``.

    Kept separate from ``parser`` construction so the ``relicllm bench`` subcommand can add these onto
    its own parser (with its launch flags beside them) while a standalone ``python -m`` invocation
    builds a parser around them. The names and defaults are vLLM's, per ``docs/guides/latency_metrics.md``.

    ``suppress_defaults`` is what makes a scenario overridable. When it is set, a flag the operator did
    **not** name is absent from the namespace rather than carrying its default, so the ``bench`` command
    can read "did they ask for this?" from presence -- exactly the question ``argparse.SUPPRESS``
    answers. Without it, a scenario's value and a flag's default are the same thing on the namespace and
    an explicit ``--num-prompts 2`` loses to the scenario's count. ``--base-url`` is the exception: it
    is bench's launch decision, not a workload field, so it is always present.
    """
    def default(value: Any) -> Any:
        return argparse.SUPPRESS if suppress_defaults else value

    parser.add_argument("--base-url", default=None, help="An already-running server to measure; omitted, the command launches one.")
    parser.add_argument("--endpoint", default=default("/v1/completions"), help="API endpoint.")
    parser.add_argument("--model", default=default(None), help="Model id to send; discovered from /v1/models when omitted.")
    parser.add_argument("--max-concurrency", type=int, default=default(None), help="Maximum number of in-flight requests.")
    parser.add_argument("--request-rate", type=float, default=default(float("inf")), help="Requests per second; inf sends all at once.")
    parser.add_argument("--burstiness", type=float, default=default(1.0), help="Gamma shape of the arrival intervals; 1 is Poisson.")
    parser.add_argument("--num-warmups", type=int, default=default(0), help="Discarded requests sent before the measured run.")
    parser.add_argument("--percentile-metrics", default=default(None), help=f'Comma-separated metrics to report percentiles for (default "{DEFAULT_PERCENTILE_METRICS}").')
    parser.add_argument("--metric-percentiles", default=default("99"), help='Comma-separated percentiles (default "99").')
    parser.add_argument("--goodput", nargs="+", default=default(None), help='Service level objectives as "KEY:VALUE" pairs in milliseconds; keys are ttft, tpot, e2el.')
    parser.add_argument("--tokenizer", default=default(None), help="Tokenizer path used to count generated tokens when the server sends no usage chunk.")
    parser.add_argument("--seed", type=int, default=default(0), help="Seed for the arrival schedule and the random dataset.")
    parser.add_argument("--timeout", type=float, default=default(1200.0), help="Per-request HTTP timeout in seconds.")
    parser.add_argument("--extra-body", default=default(None), help="JSON object merged into every request payload.")
    parser.add_argument("--max-workers-cap", type=int, default=default(1024), help="Upper bound on client threads.")
    parser.add_argument("--dataset-name", default=default("random"), choices=["random", "custom"], help="Prompt source.")
    parser.add_argument("--random-input-len", type=int, default=default(1024), help="Mean input length in tokens (random dataset).")
    parser.add_argument("--random-output-len", type=int, default=default(128), help="Mean output length in tokens.")
    parser.add_argument("--random-range-ratio", type=float, default=default(0.0), help="Sample lengths from len*(1-r)..len*(1+r).")
    parser.add_argument("--num-prompts", type=int, default=default(1000), help="Number of requests to issue.")
    parser.add_argument("--chars-per-token", type=float, default=default(4.0), help="Words-to-length conversion for the random dataset.")
    # RelicLLM extensions: not in vLLM, both in the guide.
    parser.add_argument("--token-latch", default=default("content"), choices=["content", "first-chunk"],
                        help="What counts as the first token: content (honest, default) or first-chunk (vLLM's client).")
    parser.add_argument("--no-stream", dest="stream", action="store_false", default=default(True), help="Use blocking requests; ITL and TPOT are then undefined.")


def build_parser() -> argparse.ArgumentParser:
    """The client's own parser, for ``python -m relicllm.bench.client``."""
    parser = argparse.ArgumentParser(
        description="Measure an OpenAI-compatible server on vLLM's terms. The server must already be running.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_client_arguments(parser)
    parser.set_defaults(base_url=DEFAULT_BASE_URL)
    parser.add_argument("--json-out")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    record = run_client(args)
    if args.json_out:
        pathlib.Path(args.json_out).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())