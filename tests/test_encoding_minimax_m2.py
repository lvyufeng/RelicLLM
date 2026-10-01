from __future__ import annotations

from pathlib import Path

import pytest

from src.encoding.minimax_m2 import (
    decode_minimax_m2_ids,
    encode_minimax_m2_prompt,
    minimax_m2_context_info,
    render_minimax_m2_chat_prompt,
)

REAL_MINIMAX_PATH = Path("/mnt/data1/dsv4_inference/gguf_hfd/MiniMax-M2.7-GGUF/UD-IQ1_M")


def test_minimax_chat_prompt_compact_framing() -> None:
    prompt = render_minimax_m2_chat_prompt("请用一句话介绍你自己。")

    assert prompt.startswith("]~!b[]~b]system\n")
    assert "]~b]user\n请用一句话介绍你自己。" in prompt
    assert prompt.endswith("]~b]ai\n")
    assert "<think>" not in prompt


def test_minimax_chat_prompt_can_start_thinking() -> None:
    prompt = render_minimax_m2_chat_prompt("hello", thinking=True)

    assert prompt.endswith("]~b]ai\n<think>\n")


@pytest.mark.skipif(not REAL_MINIMAX_PATH.exists(), reason="local MiniMax-M2.7 GGUF bundle not present")
def test_real_minimax_context_info_reports_192k() -> None:
    info = minimax_m2_context_info(REAL_MINIMAX_PATH)

    assert info["architecture"] == "minimax-m2"
    assert info["context_length"] == 196608
    assert info["n_layers"] == 62
    assert info["n_kv_heads"] == 8
    assert info["head_dim"] == 128
    assert info["eos_token_id"] == 200020
    assert info["kv_cache_bytes_per_token_fp16"] == 62 * 8 * 128 * 2 * 2
    assert info["kv_cache_mib_at_context_fp16"] > 45000.0


@pytest.mark.skipif(not REAL_MINIMAX_PATH.exists(), reason="local MiniMax-M2.7 GGUF bundle not present")
def test_real_minimax_tokenizer_encodes_chinese_chat_prompt() -> None:
    ids, prompt_text, metadata = encode_minimax_m2_prompt(
        REAL_MINIMAX_PATH,
        "请用一句话介绍你自己。",
        chat=True,
    )

    assert prompt_text.endswith("]~b]ai\n")
    assert ids[:2] == [200034, 200019]
    assert 200020 in ids
    assert int(metadata["tokenizer.ggml.eos_token_id"]) == 200020


@pytest.mark.skipif(not REAL_MINIMAX_PATH.exists(), reason="local MiniMax-M2.7 GGUF bundle not present")
def test_real_minimax_tokenizer_decodes_generated_ids() -> None:
    text = decode_minimax_m2_ids(
        REAL_MINIMAX_PATH,
        [758, 3100, 20886, 58, 494, 4088, 829, 49864, 10201, 60103],
    )

    assert "The user asks" in text
    assert "请用一句话介绍你自己" in text
