"""The DeepSeek-V4 answer builder, on the parts the HTTP route reads.

No checkpoint and no device: `_format_completion_result` is the function that turns a run's ids
into the mapping the torch adapter reads, and both tests here pin one key of it.
"""

from __future__ import annotations

from relicllm.models.deepseek_v4.serving import _format_completion_result


class _Tokenizer:
    """One character per id, so the decoded text of `[2, 3]` is `"bc"` and needs no real vocab."""

    _TEXT = {2: "b", 3: "c", 4: "d"}

    def decode(self, token_ids, **_kwargs) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return "".join(self._TEXT.get(int(token), "?") for token in token_ids)


def _result(**kwargs):
    return _format_completion_result(
        _Tokenizer(), "chat", [1], [2, 3], 0.1, 0.2, 1, 2, **kwargs
    )


def test_a_ranking_is_rendered_as_the_openai_logprobs_object():
    """The HTTP route reads `result["logprobs"]`; a bare float list there is the wrong shape.

    The model-side payload builder is the one that knows how to shape it -- the same builder the
    legacy response edge used -- and it was bypassed for the adapter route, which fell back to
    `token_logprobs`. Both spellings are still on the result: the list is what the payload builder
    reads from.
    """
    result = _result(
        logprobs=[
            {"token_id": 2, "logprob": -0.5, "top_logprobs": [{"token_id": 4, "logprob": -1.25}]},
            {"token_id": 3, "logprob": -0.75, "top_logprobs": []},
        ],
        top_logprobs=1,
    )

    assert result["token_logprobs"] == [-0.5, -0.75]
    assert result["logprobs"]["content"] == [
        {
            "token": "b",
            "logprob": -0.5,
            "bytes": [98],
            "top_logprobs": [{"token": "d", "logprob": -1.25, "bytes": [100]}],
        },
        # An entry with no alternatives carries no `top_logprobs` key at all -- the builder adds it
        # only when there are candidates, and an empty list would read as "ranked none".
        {"token": "c", "logprob": -0.75, "bytes": [99]},
    ]


def test_a_request_that_asked_for_no_ranking_carries_no_logprobs_key():
    """The field is absent rather than null, which is what a client reads as "not asked for"."""
    result = _result()

    assert "logprobs" not in result
    assert "token_logprobs" not in result
