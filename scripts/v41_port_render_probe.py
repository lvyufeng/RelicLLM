"""Print the ids the *port's* chat renderer feeds, so they can be diffed against the derived ones.

`V41Backend._encode_chat` renders a user turn by calling the checkpoint's own `encode_messages`
through `relicllm.encoding.deepseek_v4_1` and then tokenizing the rendered text with
`add_special_tokens=False`. `scripts/v41_templated_ids_probe.py` derives the same ids from the
checkpoint's encoder and tokenizer alone, without importing the port. Running both and comparing the
ints is what makes "the port and the reference saw the same prompt" a measurement instead of an
inference from a token count -- a count of 13 is also what two *different* 13-token prompts would
produce.

CPU-only on purpose: no card is needed to render a prompt, and the check is worth running even when
every GPU is busy.

    PYTHONPATH=<relicllm checkout> python scripts/v41_port_render_probe.py --checkpoint <ckpt>

Do not pass `-I`: it implies `-E`, which drops `PYTHONPATH`, and the run then renders through
whichever tree the environment's editable install points at rather than the one named above.
"""

from __future__ import annotations

import argparse
import json

DEFAULT_PROMPT = "Reply with the single line: golden fixture."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--thinking-mode", default="chat", choices=["chat", "thinking"])
    args = parser.parse_args(argv)

    import relic_core
    from transformers import AutoTokenizer

    import relicllm
    from relicllm.encoding.deepseek_v4_1 import encode_messages

    text = encode_messages(
        args.checkpoint,
        [{"role": "user", "content": args.prompt}],
        thinking_mode=args.thinking_mode,
        reasoning_effort_value=None,
        tools=None,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    # The spelling `_encode_chat` uses; the rendered text opens with its own beginning-of-sentence
    # token, so the tokenizer must not add a second one.
    ids = [int(t) for t in tokenizer(text, add_special_tokens=False)["input_ids"]]
    print(json.dumps({
        "relicllm": relicllm.__file__,
        "relic_core": relic_core.__file__,
        "thinking_mode": args.thinking_mode,
        "n": len(ids),
        "ids": ids,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())