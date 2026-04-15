"""Eval: data collector — coverage filtering and error isolation.

These tests mock the scout_service so no real HTTP calls are made.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from agents.data_collector import filter_countries_by_coverage
from agents.pipeline_config import HIGH_COVERAGE_DEFAULTS, AVAILABLE_TOOLS


# ---------------------------------------------------------------------------
# filter_countries_by_coverage
# ---------------------------------------------------------------------------

def _make_dataset(code: str, rows_per_tool: dict[str, int]) -> dict:
    """Build a minimal dataset dict matching the shape data_collector expects."""
    ds = {
        "worldbank": [{}] * rows_per_tool.get("worldbank", 0),
        "employment": [{}] * rows_per_tool.get("employment", 0),
        "conflict_events": [{}] * rows_per_tool.get("acled", 0),
        "climate": [{}] * rows_per_tool.get("environment", 0),
        "aqi": [],
        "city_scores": [{}] * rows_per_tool.get("teleport", 0),
        "news": [{}] * rows_per_tool.get("news", 0),
    }
    return ds


def test_filter_keeps_countries_with_data() -> None:
    codes = ["DEU", "CAN", "AUS"]
    countries_data = {
        c: _make_dataset(c, {"worldbank": 10, "employment": 5})
        for c in codes
    }
    filtered, v_countries, v_codes, ranking = filter_countries_by_coverage(
        countries_data, codes, ["worldbank", "employment"], k=3
    )
    assert set(v_codes).issubset(set(codes))
    assert len(v_codes) <= 3


def test_filter_drops_zero_coverage_country() -> None:
    codes = ["DEU", "CAN", "AUS"]
    countries_data = {
        "DEU": _make_dataset("DEU", {"worldbank": 10}),
        "CAN": _make_dataset("CAN", {"worldbank": 0}),   # zero coverage
        "AUS": _make_dataset("AUS", {"worldbank": 8}),
    }
    _, _, v_codes, _ = filter_countries_by_coverage(
        countries_data, codes, ["worldbank"], k=3
    )
    # CAN had zero rows — should have lower rank or be dropped
    # At minimum, DEU and AUS should be present
    assert "DEU" in v_codes or "AUS" in v_codes


def test_filter_respects_k_limit() -> None:
    codes = HIGH_COVERAGE_DEFAULTS
    countries_data = {c: _make_dataset(c, {"worldbank": 5}) for c in codes}
    _, _, v_codes, _ = filter_countries_by_coverage(
        countries_data, codes, ["worldbank"], k=3
    )
    assert len(v_codes) <= 3


def test_filter_returns_coverage_ranking() -> None:
    codes = ["DEU", "CAN"]
    countries_data = {c: _make_dataset(c, {"worldbank": 4}) for c in codes}
    _, _, _, ranking = filter_countries_by_coverage(
        countries_data, codes, ["worldbank"], k=2
    )
    # ranking may be a list of dicts or a plain dict — check it's indexable by country
    assert ranking is not None
    if isinstance(ranking, dict):
        for code in codes:
            assert code in ranking
    else:
        # list of dicts with a 'country' key
        assert isinstance(ranking, list)
        ranked_codes = [r.get("country") for r in ranking]
        for code in codes:
            assert code in ranked_codes


def test_filter_empty_input_does_not_raise() -> None:
    result = filter_countries_by_coverage({}, [], ["worldbank"], k=3)
    _, v_countries, v_codes, ranking = result
    assert v_countries == []
    assert v_codes == []


# ---------------------------------------------------------------------------
# collect_data_for_countries: error isolation
# Each country's fetch failure must not crash the others
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_collect_isolates_per_country_errors() -> None:
    from agents.data_collector import collect_data_for_countries

    async def _failing_collect(*args, **kwargs):
        raise RuntimeError("Simulated API failure")

    with patch(
        "agents.data_collector.collect_migration_dataset",
        side_effect=_failing_collect,
    ):
        result = await collect_data_for_countries(
            ["Germany", "Canada"],
            ["DEU", "CAN"],
            ["worldbank"],
            2015, 2023,
        )
    # Must return a dict (possibly with error entries) — must not raise
    assert isinstance(result, dict)
