"""Run the port's *served* chat path over the checkpoint's chat template and dump the answer.

The golden fixtures and both oracle recorders compare on the **raw** completion prompt, but the
checkpoint's own `inference/generate.py` renders through `encode_messages` (the chat template) before
it forwards. For an instruct model those are different questions, and the templated one is the one the
official entry point actually asks. This probe drives `relicllm`'s chat path -- the same one `serve`
uses -- through the canonical fixture's own argv, so its answer is directly comparable to a
chat-templated reference recording.

    PYTHONPATH=<relicllm checkout> python scripts/v41_chat_parity_probe.py \
        --model <ckpt> --out /tmp/port_chat.pt

Do not add `-I`: it implies `-E`, which drops `PYTHONPATH`, and the run then imports
whichever tree the environment's editable install points at rather than the one named above.
A probe that reports numbers from a revision nobody chose reads exactly like one that works.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

#: The golden fixture this probe takes its argv from, so the two cannot drift apart.
GOLDEN_FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "golden" / "v41.json"


def main() -> int:
    fixture = json.loads(GOLDEN_FIXTURE.read_text(encoding="utf-8"))

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default="/tmp/port_chat.pt")
    parser.add_argument("--prompt", default="Reply with the single line: golden fixture.")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=8)
    # Everything after `--` replaces the fixture's argv tail. The default is the fixture's own
    # canonical TP4 set (4x sm_89 cards); an sm_75 box that cannot hold a 4-way split passes its own,
    # e.g. `-- --max-model-len 8192 --backend-option prefill_chunk=4096 --backend-option threads=22`.
    parser.add_argument(
        "extra",
        nargs="*",
        help="argv tail after `--`; empty means the canonical TP4 set",
    )
    args = parser.parse_args()

    # The logits record is what the backend emits when this is set; the same knob the golden harnesses
    # read. `_logits_check_top_k` reads it at request time, so setting it here is soon enough.
    os.environ["POCKETLLM_V41_LOGITS_CHECK"] = str(args.top_k)
    os.environ.setdefault("DEEPSEEK_V41_RESIDENT_EXPERTS", "1")

    import torch

    from relicllm import LLM
    from relicllm.api import SamplingParams
    from relicllm.cli import _args, build_parser

    # The canonical fixture's own argv, read from the fixture file rather than copied so a fixture
    # edit reaches this probe; only the checkpoint is re-pointed, at `--model`.
    if args.extra:
        argv = ["serve", "--backend", "v41", "--model", args.model, *args.extra]
    else:
        argv = list(fixture["argv"])
        argv[argv.index("--model") + 1] = args.model
    namespace = build_parser().parse_args(argv)

    def say(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    llm = LLM(_args(namespace))
    try:
        result = llm.chat(
            [{"role": "user", "content": args.prompt}],
            SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
        )[0]
        logits_check = result.metadata.get("logits_check")
        print("TOKENS", list(result.token_ids), flush=True)
        print("TEXT", repr(result.text), flush=True)
        print("PROMPT_TOKENS", getattr(result.usage, "prompt_tokens", None), flush=True)
        print("LOGITS_CHECK", json.dumps(logits_check), flush=True)
        torch.save(
            {
                "tokens": list(result.token_ids),
                "text": result.text,
                "prompt_tokens": getattr(result.usage, "prompt_tokens", None),
                "logits_check": logits_check,
            },
            args.out,
        )
        say(f"wrote {args.out}")
    finally:
        llm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())