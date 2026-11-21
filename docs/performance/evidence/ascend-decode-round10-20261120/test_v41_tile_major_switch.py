"""The loader's tile-major expert switch, checked where it can be checked without a card.

The switch selects which NPU packed-expert store the loader builds. It is off by default, and the
optional store is imported only when it is on -- a regression that referenced that lazily imported
class from the default path made a full-model run die at load with no Python traceback, while the
store's own test suite stayed green because it constructs the stores directly instead of going
through this branch. These tests pin the parsing and the default, which is the part a card would
otherwise be needed to notice.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from relicllm.models.deepseek_v4_1.loader import (  # noqa: E402
    TILE_MAJOR_EXPERTS_ENV,
    tile_major_experts_enabled,
)


@pytest.mark.parametrize("value", [None, "0", "false", "no", "off", "", "2", "enabled"])
def test_anything_but_an_explicit_truthy_value_keeps_the_default_off(monkeypatch, value):
    """Unset is the historical path, and a typo must not be read as "on"."""
    if value is None:
        monkeypatch.delenv(TILE_MAJOR_EXPERTS_ENV, raising=False)
    else:
        monkeypatch.setenv(TILE_MAJOR_EXPERTS_ENV, value)
    assert tile_major_experts_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " On "])
def test_an_explicit_truthy_value_turns_it_on(monkeypatch, value):
    monkeypatch.setenv(TILE_MAJOR_EXPERTS_ENV, value)
    assert tile_major_experts_enabled() is True


def test_the_optional_store_is_not_imported_by_the_loader_itself():
    """The tile-major class must stay a lazy import, or selecting it is meaningless."""
    from relicllm.models.deepseek_v4_1 import loader

    assert not hasattr(loader, "AscendTileMajorRoutedExperts"), (
        "loader imports the tile-major store eagerly; it is supposed to be imported only inside the "
        "branch that selects it, so that the default path never depends on it"
    )
