#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
from urllib import request


def _post_json(url: str, body: dict, timeout: float = 3600.0) -> dict:
    req = request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_json(url: str, timeout: float = 10.0) -> dict:
    with request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _run_case(url: str, prompt: str, max_tokens: int, model: str) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    t0 = time.perf_counter()
    resp = _post_json(url, body)
    wall_time = time.perf_counter() - t0
    timings = resp.get("deepseek_timings", {})
    usage = resp.get("usage", {})
    message = ((resp.get("choices") or [{}])[0].get("message") or {})
    return {
        "wall_time": wall_time,
        "prefill_time": timings.get("prefill_time"),
        "decode_time": timings.get("decode_time"),
        "prefill_tokens": timings.get("prefill_tokens"),
        "decode_tokens": timings.get("decode_tokens"),
        "decode_tokens_per_second": timings.get("decode_tokens_per_second"),
        "prefill_tokens_per_second": timings.get("prefill_tokens_per_second"),
        "ttft": timings.get("ttft"),
        "ttft_ms": timings.get("ttft_ms"),
        "tpot": timings.get("tpot"),
        "tpot_ms": timings.get("tpot_ms"),
        "throughput_tokens_per_second": timings.get("throughput_tokens_per_second"),
        "total_tokens_per_second": timings.get("total_tokens_per_second"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "content": message.get("content", ""),
    }


def _load_prompt(path: str) -> str:
    return Path(path).read_text().strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--short-input", required=True)
    parser.add_argument("--long-input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case", choices=["all", "short_short", "short_long", "long_short", "long_long"], default="all")
    parser.add_argument("--warmup", choices=["none", "long_long"], default="none")
    parser.add_argument("--short-max-tokens", type=int, default=8)
    parser.add_argument("--long-max-tokens", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1")

    short_prompt = _load_prompt(args.short_input)
    long_prompt = _load_prompt(args.long_input)
    health_url = args.base_url.rstrip("/") + "/health"
    chat_url = args.base_url.rstrip("/") + "/v1/chat/completions"

    for _ in range(1800):
        try:
            _get_json(health_url, timeout=2.0)
            break
        except Exception:
            time.sleep(1.0)
    else:
        raise RuntimeError(f"server did not become healthy: {health_url}")

    results = {
        "base_url": args.base_url,
        "model": args.model,
        "warmup": None,
        "cases": {},
    }
    if args.warmup == "long_long":
        results["warmup"] = _run_case(chat_url, long_prompt, args.long_max_tokens, args.model)

    cases = {
        "short_short": (short_prompt, args.short_max_tokens),
        "short_long": (short_prompt, args.long_max_tokens),
        "long_short": (long_prompt, args.short_max_tokens),
        "long_long": (long_prompt, args.long_max_tokens),
    }
    selected = list(cases) if args.case == "all" else [args.case]
    for name in selected:
        prompt, max_tokens = cases[name]
        case_results = []
        for idx in range(args.repeat):
            result = _run_case(chat_url, prompt, max_tokens, args.model)
            case_results.append(result)
            prefill_tps = result.get("prefill_tokens_per_second")
            if prefill_tps is None and result.get("prefill_tokens") is not None and result.get("prefill_time") is not None:
                prefill_tps = result["prefill_tokens"] / max(result["prefill_time"], 1e-9)
            tpot_ms = result.get("tpot_ms")
            if tpot_ms is None and result.get("decode_time") is not None and result.get("decode_tokens") is not None:
                tpot_ms = result["decode_time"] * 1000.0 / max(result["decode_tokens"], 1)
            throughput_tps = result.get("throughput_tokens_per_second") or result.get("total_tokens_per_second")
            if throughput_tps is None and result.get("prefill_time") is not None and result.get("decode_time") is not None:
                throughput_tps = (result.get("prefill_tokens") or 0) + (result.get("decode_tokens") or 0)
                throughput_tps /= max(result["prefill_time"] + result["decode_time"], 1e-9)
            print(
                f"{name}[{idx + 1}/{args.repeat}]: wall={result['wall_time']:.3f}s "
                f"prefill={result['prefill_time']:.3f}s prefill_tps={prefill_tps:.3f} "
                f"decode={result['decode_time']:.3f}s decode_tps={result['decode_tokens_per_second']:.3f} "
                f"ttft={result.get('ttft', result['prefill_time']):.3f}s tpot={tpot_ms:.3f}ms throughput={throughput_tps:.3f}",
                flush=True,
            )
        best = max(case_results, key=lambda item: item.get("decode_tokens_per_second") or 0.0)
        mean_tps = sum((item.get("decode_tokens_per_second") or 0.0) for item in case_results) / len(case_results)
        results["cases"][name] = {
            "runs": case_results,
            "best": best,
            "mean_decode_tokens_per_second": mean_tps,
        }
        if args.repeat > 1:
            print(
                f"{name}: best_decode_tps={best['decode_tokens_per_second']:.3f} "
                f"mean_decode_tps={mean_tps:.3f}",
                flush=True,
            )

    Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
