from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from tokenizers import Tokenizer

from src.encoding.gguf_tokenizer import (
    build_gguf_bpe_tokenizer,
    context_length_from_metadata,
    decode_ids,
    metadata_int,
    parse_ids_csv,
    read_gguf_metadata,
)

MINIMAX_M2_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Your name is MiniMax-M2.7 and is built by MiniMax."


def build_minimax_m2_tokenizer(gguf_path: str | Path) -> tuple[Tokenizer, dict[str, Any]]:
    tokenizer, metadata = build_gguf_bpe_tokenizer(gguf_path)
    architecture = metadata.get("general.architecture")
    if architecture not in (None, "", "minimax-m2"):
        raise ValueError(f"expected minimax-m2 GGUF tokenizer metadata, got architecture={architecture!r}")
    return tokenizer, metadata


def render_minimax_m2_chat_prompt(
    user_text: str,
    *,
    system_prompt: str = MINIMAX_M2_DEFAULT_SYSTEM_PROMPT,
    thinking: bool = False,
) -> str:
    """Render a compact MiniMax-M2 chat prompt for smoke generation.

    The GGUF metadata includes a full Jinja chat template for production serving.
    This compact renderer intentionally covers the common system/user/assistant
    framing used by local raw-token generation smoke tests. ``thinking=True``
    appends the model's think opener; by default generation starts at visible
    assistant text.
    """

    prompt = (
        "]~!b[]~b]system\n"
        + system_prompt
        + "[e~[\n"
        + "]~b]user\n"
        + user_text
        + "[e~[\n"
        + "]~b]ai\n"
    )
    if thinking:
        prompt += "<think>\n"
    return prompt


def encode_minimax_m2_prompt(
    gguf_path: str | Path,
    prompt: str,
    *,
    chat: bool = False,
    thinking: bool = False,
    system_prompt: str = MINIMAX_M2_DEFAULT_SYSTEM_PROMPT,
) -> tuple[list[int], str, dict[str, Any]]:
    tokenizer, metadata = build_minimax_m2_tokenizer(gguf_path)
    prompt_text = (
        render_minimax_m2_chat_prompt(prompt, system_prompt=system_prompt, thinking=thinking)
        if chat
        else prompt
    )
    encoded = tokenizer.encode(prompt_text, add_special_tokens=False)
    return [int(token_id) for token_id in encoded.ids], prompt_text, metadata


def decode_minimax_m2_ids(
    gguf_path: str | Path,
    ids: Iterable[int],
    *,
    skip_special: bool = False,
) -> str:
    tokenizer, _metadata = build_minimax_m2_tokenizer(gguf_path)
    return decode_ids(tokenizer, ids, skip_special=skip_special)


def minimax_m2_context_info(gguf_path: str | Path) -> dict[str, Any]:
    metadata = read_gguf_metadata(gguf_path, keep_array_keys=(), keep_all_scalars=True)
    context_length = context_length_from_metadata(metadata, architecture="minimax-m2")
    n_layers = metadata_int(metadata, "minimax-m2.block_count", 0)
    n_kv_heads = metadata_int(metadata, "minimax-m2.attention.head_count_kv", 0)
    head_dim = metadata_int(metadata, "minimax-m2.attention.key_length", 0)
    dtype_bytes = 2
    kv_bytes_per_token = n_layers * n_kv_heads * head_dim * 2 * dtype_bytes
    return {
        "architecture": metadata.get("general.architecture"),
        "context_length": int(context_length),
        "n_layers": int(n_layers),
        "n_kv_heads": int(n_kv_heads),
        "head_dim": int(head_dim),
        "kv_cache_bytes_per_token_fp16": int(kv_bytes_per_token),
        "kv_cache_mib_at_context_fp16": (float(kv_bytes_per_token) * float(context_length) / 1024.0 / 1024.0)
        if context_length and kv_bytes_per_token
        else 0.0,
        "bos_token_id": metadata_int(metadata, "tokenizer.ggml.bos_token_id", -1),
        "eos_token_id": metadata_int(metadata, "tokenizer.ggml.eos_token_id", -1),
        "unknown_token_id": metadata_int(metadata, "tokenizer.ggml.unknown_token_id", -1),
        "padding_token_id": metadata_int(metadata, "tokenizer.ggml.padding_token_id", -1),
    }


def _encode_payload(
    gguf_path: str | Path,
    prompt: str,
    *,
    chat: bool,
    thinking: bool,
    system_prompt: str,
) -> dict[str, Any]:
    ids, prompt_text, metadata = encode_minimax_m2_prompt(
        gguf_path,
        prompt,
        chat=chat,
        thinking=thinking,
        system_prompt=system_prompt,
    )
    return {
        "prompt_text": prompt_text,
        "prompt_ids": ids,
        "prompt_csv": ",".join(str(token_id) for token_id in ids),
        "context_length": context_length_from_metadata(metadata, architecture="minimax-m2"),
        "eos_token_id": metadata_int(metadata, "tokenizer.ggml.eos_token_id", -1),
        "bos_token_id": metadata_int(metadata, "tokenizer.ggml.bos_token_id", -1),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MiniMax-M2 GGUF tokenizer encode/decode helper")
    parser.add_argument("--gguf", required=True, help="GGUF file, shard, or directory")
    parser.add_argument("--prompt", help="Text prompt to encode")
    parser.add_argument("--chat", action="store_true", help="Wrap --prompt in compact MiniMax-M2 chat framing")
    parser.add_argument("--thinking", action="store_true", help="With --chat, append <think> before generation")
    parser.add_argument("--system-prompt", default=MINIMAX_M2_DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--ids", help="Comma-separated token ids to decode")
    parser.add_argument("--ids-file", help="File containing comma-separated token ids to decode")
    parser.add_argument("--skip-special", action="store_true", help="Skip special tokens while decoding")
    parser.add_argument("--context-info", action="store_true", help="Print MiniMax-M2 context metadata summary")
    parser.add_argument("--json", action="store_true", help="Emit JSON for encode/context output")
    args = parser.parse_args(argv)

    if args.context_info:
        info = minimax_m2_context_info(args.gguf)
        if args.json:
            print(json.dumps(info, ensure_ascii=False))
        else:
            for key, value in info.items():
                print(f"{key}={value}")

    if args.prompt is not None:
        payload = _encode_payload(
            args.gguf,
            args.prompt,
            chat=args.chat,
            thinking=args.thinking,
            system_prompt=args.system_prompt,
        )
        if args.json:
            print(json.dumps(payload, ensure_ascii=False))
        else:
            print(payload["prompt_csv"])

    ids_text = args.ids
    if args.ids_file:
        ids_text = Path(args.ids_file).read_text(encoding="utf-8")
    if ids_text:
        print(decode_minimax_m2_ids(args.gguf, parse_ids_csv(ids_text), skip_special=args.skip_special))


if __name__ == "__main__":
    main()
