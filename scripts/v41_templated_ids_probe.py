"""Derive the ids the checkpoint's own entry point feeds a chat turn, without asking the port.

The committed v41 oracle is a *raw* completion prompt; `inference/generate.py` -- the checkpoint's
own entry point -- renders a user turn through `encode_messages` first, at its line 156:

    prompt_tokens = tokenizer.encode(encode_messages(messages, thinking_mode=thinking_mode))

with `--thinking-mode` defaulting to `chat`. A chat-templated recording has to be pinned to the ids
that call produces, and the only way to know them is to run it: the template is a Python module
under `<checkpoint>/encoding/`, and reading it is not the same as executing it.

This probe is deliberately **not** built on `relicllm`. The port has its own renderer
(`V41Backend._encode_chat`, which calls the same checkpoint file and then tokenizes the rendered
text with `add_special_tokens=False` because the text opens with its own beginning-of-sentence
token). Deriving the ids through the port would make the oracle's input a statement about the port;
deriving them from the checkpoint's own code makes them a statement about the checkpoint, and the
two are then two independently produced numbers that can disagree.

Both tokenizer spellings are printed, because the two sides of the comparison really do use
different ones and a mismatch between them would be invisible in a token count or a decoded string:

- `generate.py` lets the tokenizer add special tokens (the default);
- the port passes `add_special_tokens=False` over the rendered text.

Usage (on a host that holds the checkpoint)::

    python scripts/v41_templated_ids_probe.py --checkpoint <ckpt> [--prompt ...] [--out FILE]

Prints one JSON object, so a caller can diff it against a recording instead of reading it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from typing import Any

#: The raw prompt, kept in step with the two recorders' `DEFAULT_PROMPT` so the chat rendering of
#: this string and the raw ids they pin are two renderings of one input.
DEFAULT_PROMPT = "Reply with the single line: golden fixture."

#: The directory and file the released checkpoint keeps its encoder in -- the same names
#: `relicllm.encoding.deepseek_v4_1` uses, restated here because this file must not import it.
ENCODER_DIRNAME = "encoding"
ENCODER_FILENAME = "encoding.py"


def load_encoder(checkpoint_dir: str):
    """Import the checkpoint's own `encoding/encoding.py`, by path.

    By path rather than by putting the checkpoint on `sys.path`: a checkpoint is data, and a data
    directory that lands on the import path can shadow any module name it happens to contain.
    """
    path = os.path.join(checkpoint_dir, ENCODER_DIRNAME, ENCODER_FILENAME)
    if not os.path.isfile(path):
        raise SystemExit(f"{path} does not exist; is {checkpoint_dir} a V4.1 checkpoint?")
    spec = importlib.util.spec_from_file_location("_v41_encoding_probe", path)
    if spec is None or spec.loader is None:  # pragma: no cover - a readable file always has one
        raise SystemExit(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="the V4.1 checkpoint directory")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--thinking-mode",
        default="chat",
        choices=["chat", "thinking"],
        help="`generate.py`'s --thinking-mode; `chat` is its default",
    )
    parser.add_argument("--out", help="also write the result as JSON here")
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    encoder = load_encoder(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    messages = [{"role": "user", "content": args.prompt}]

    rendered = encoder.encode_messages(messages, thinking_mode=args.thinking_mode)
    # generate.py's own spelling: the default lets the tokenizer add special tokens.
    ids_add_special = [int(t) for t in tokenizer.encode(rendered)]
    # The port's spelling: the rendered text already opens with the beginning-of-sentence token,
    # so the tokenizer must not add another.
    ids_plain = [int(t) for t in tokenizer(rendered, add_special_tokens=False)["input_ids"]]

    result: dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "prompt": args.prompt,
        "thinking_mode": args.thinking_mode,
        "n": len(ids_plain),
        "ids": ids_plain,
        "ids_add_special_tokens": ids_add_special,
        "spellings_agree": ids_add_special == ids_plain,
        "rendered_text": rendered,
        "decoded": tokenizer.decode(ids_plain),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())