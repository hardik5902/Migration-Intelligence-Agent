"""Eval: EDA analyst — series extraction, growth rates, anomaly detection.

Uses the synthetic fixtures from conftest.py.  No real API calls.
"""

from __future__ import annotations

import pytest

from agents.eda_analyst import (
    _ALL_METRICS,
    _TOOL_METRICS,
    _TOOL_ANCHOR,
    run_eda,
)
from agents.pipeline_config import AVAILABLE_TOOLS


# ---------------------------------------------------------------------------
# _ALL_METRICS integrity
# ---------------------------------------------------------------------------

def test_all_metrics_is_non_empty() -> None:
    assert _ALL_METRICS, "_ALL_METRICS must not be empty"


def test_every_metric_has_extractor_and_label() -> None:
    for metric, entry in _ALL_METRICS.items():
        assert callable(entry[0]), f"Extractor for {metric!r} is not callable"
        assert isinstance(entry[1], str) and entry[1], (
            f"Label for {metric!r} must be a non-empty string"
        )


def test_no_displacement_in_all_metrics() -> None:
    assert "displacement" not in _ALL_METRICS, (
        "displacement metric was not removed from _ALL_METRICS"
    )


# ---------------------------------------------------------------------------
# _TOOL_METRICS coverage
# ---------------------------------------------------------------------------

def test_tool_metrics_keys_match_available_tools() -> None:
    active_tools = set(AVAILABLE_TOOLS.keys()) - {"news"}   # news has no EDA metrics
    for tool in active_tools:
        assert tool in _TOOL_METRICS, f"Tool {tool!r} missing from _TOOL_METRICS"


def test_no_unhcr_in_tool_metrics() -> None:
    assert "unhcr" not in _TOOL_METRICS


@pytest.mark.parametrize("tool", list(_TOOL_METRICS.keys()))
def test_every_tool_metric_is_registered(tool: str) -> None:
    for m in _TOOL_METRICS[tool]:
        assert m in _ALL_METRICS, (
            f"Metric {m!r} listed under _TOOL_METRICS[{tool!r}] "
            f"but not present in _ALL_METRICS"
        )


# ---------------------------------------------------------------------------
# _TOOL_ANCHOR: anchor must point to a real metric
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tool", list(_TOOL_ANCHOR.keys()))
def test_tool_anchor_is_valid_metric(tool: str) -> None:
    anchor = _TOOL_ANCHOR[tool]
    assert anchor in _ALL_METRICS, (
        f"_TOOL_ANCHOR[{tool!r}] = {anchor!r} is not in _ALL_METRICS"
    )


# ---------------------------------------------------------------------------
# run_eda smoke test with synthetic data
# ---------------------------------------------------------------------------

def test_run_eda_returns_expected_keys(two_country_dataset: dict) -> None:
    result = run_eda(
        two_country_dataset,
        selected_tools=["worldbank", "employment"],
        query_focus="economic conditions",
        worldbank_indicators=["gdp_growth", "inflation"],
    )
    for key in ("findings", "growth_rates", "anomalies", "stats_summary",
                "latest_matrix", "active_metrics_ordered"):
        assert key in result, f"run_eda result missing key: {key!r}"


def test_run_eda_findings_are_list(two_country_dataset: dict) -> None:
    result = run_eda(
        two_country_dataset,
        selected_tools=["worldbank"],
        query_focus="health",
        worldbank_indicators=["life_expectancy"],
    )
    assert isinstance(result["findings"], list)


def test_run_eda_active_metrics_nonempty_for_known_tools(two_country_dataset: dict) -> None:
    result = run_eda(
        two_country_dataset,
        selected_tools=["worldbank", "employment"],
        query_focus="labour market",
        worldbank_indicators=["unemployment"],
    )
    assert result["active_metrics_ordered"], (
        "active_metrics_ordered should not be empty when worldbank/employment data is present"
    )


def test_run_eda_empty_dataset_does_not_raise() -> None:
    # Pass two countries with empty-but-keyed data so the matrix builder
    # doesn't receive a completely empty DataFrame.
    from agents.pipeline_config import HIGH_COVERAGE_DEFAULTS
    empty_shaped = {
        code: {"worldbank": [], "employment": [], "conflict_events": [],
               "climate": [], "aqi": [], "city_scores": [], "news": []}
        for code in HIGH_COVERAGE_DEFAULTS[:2]
    }
    result = run_eda(
        empty_shaped,
        selected_tools=["worldbank"],
        query_focus="test",
        worldbank_indicators=[],
    )
    assert "findings" in result


def test_run_eda_no_displacement_findings(two_country_dataset: dict) -> None:
    result = run_eda(
        two_country_dataset,
        selected_tools=list(AVAILABLE_TOOLS.keys()),
        query_focus="general",
        worldbank_indicators=[],
    )
    finding_titles = [f.get("title", "") for f in result.get("findings", [])]
    assert not any("displacement" in t.lower() for t in finding_titles), (
        "Displacement finding appeared after UNHCR removal"
    )
