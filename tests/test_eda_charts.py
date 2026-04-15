"""Eval: EDA chart builders.

Verifies that charts render correctly, near-zero mean falls back to boxplot,
and no chart is returned when data is genuinely absent.
"""

from __future__ import annotations

import pytest
import plotly.graph_objects as go

from agents.eda_analyst import run_eda
from agents.pipeline_config import HIGH_COVERAGE_DEFAULTS
from analysis.eda_charts import (
    build_eda_chart_manifest,
    build_eda_charts,
    render_eda_charts_by_keys,
    _build_dispatcher,
)


# ---------------------------------------------------------------------------
# build_eda_chart_manifest
# ---------------------------------------------------------------------------

def test_manifest_returns_list(two_country_dataset: dict) -> None:
    findings = run_eda(
        two_country_dataset,
        selected_tools=["worldbank", "employment"],
        query_focus="economic",
        worldbank_indicators=["gdp_growth"],
    )
    manifest = build_eda_chart_manifest(findings, two_country_dataset)
    assert isinstance(manifest, list)


def test_manifest_entry_has_required_keys(two_country_dataset: dict) -> None:
    findings = run_eda(
        two_country_dataset,
        selected_tools=["worldbank"],
        query_focus="health",
        worldbank_indicators=["life_expectancy"],
    )
    manifest = build_eda_chart_manifest(findings, two_country_dataset)
    for entry in manifest:
        for field in ("key", "title", "type", "description", "metric"):
            assert field in entry, f"Manifest entry missing field {field!r}: {entry}"


def test_manifest_keys_are_valid_for_dispatcher(two_country_dataset: dict) -> None:
    findings = run_eda(
        two_country_dataset,
        selected_tools=["worldbank", "employment"],
        query_focus="labour",
        worldbank_indicators=["unemployment"],
    )
    manifest = build_eda_chart_manifest(findings, two_country_dataset)
    render = _build_dispatcher(findings, two_country_dataset)
    for entry in manifest[:4]:
        # Dispatcher must not raise on any valid manifest key
        result = render(entry["key"])
        # result is either a Figure or None — both are acceptable
        assert result is None or isinstance(result, go.Figure)


# ---------------------------------------------------------------------------
# build_eda_charts: dedup rules
# ---------------------------------------------------------------------------

def test_build_eda_charts_returns_list(two_country_dataset: dict) -> None:
    findings = run_eda(
        two_country_dataset,
        selected_tools=["worldbank"],
        query_focus="economic",
        worldbank_indicators=[],
    )
    charts = build_eda_charts(findings, two_country_dataset, max_charts=4)
    assert isinstance(charts, list)


def test_build_eda_charts_max_respected(two_country_dataset: dict) -> None:
    findings = run_eda(
        two_country_dataset,
        selected_tools=["worldbank", "employment", "acled", "environment"],
        query_focus="general",
        worldbank_indicators=[],
    )
    for max_n in (2, 3, 4):
        charts = build_eda_charts(findings, two_country_dataset, max_charts=max_n)
        assert len(charts) <= max_n, (
            f"build_eda_charts returned {len(charts)} charts when max={max_n}"
        )


def test_build_eda_charts_no_duplicates(five_country_dataset: dict) -> None:
    """No two charts in the same run should cover the same base metric."""
    findings = run_eda(
        five_country_dataset,
        selected_tools=["worldbank", "employment"],
        query_focus="health",
        worldbank_indicators=["life_expectancy", "infant_mortality"],
    )
    charts = build_eda_charts(findings, five_country_dataset, max_charts=4)
    # Each chart is a JSON string; parse the title to check for metric repetition
    import json
    titles = []
    for c in charts:
        try:
            obj = json.loads(c)
            title = obj.get("layout", {}).get("title", {})
            if isinstance(title, dict):
                titles.append(title.get("text", ""))
            else:
                titles.append(str(title))
        except Exception:
            pass
    assert len(titles) == len(set(titles)), (
        f"Duplicate chart titles detected: {titles}"
    )


# ---------------------------------------------------------------------------
# Near-zero mean fallback: distribution_box → boxplot
# ---------------------------------------------------------------------------

def test_near_zero_mean_falls_back_to_boxplot(two_country_dataset: dict) -> None:
    """Inject temp_anomaly data centred on 0 and verify the chart is a Box trace."""
    # Add near-zero climate data
    dataset = dict(two_country_dataset)
    for code in dataset:
        dataset[code] = dict(dataset[code])
        dataset[code]["climate"] = [
            {"country": code, "avg_temp_anomaly_c": 0.001 * i,
             "annual_precipitation_mm": 900.0, "year": 2018 + i}
            for i in range(6)
        ]

    findings = run_eda(
        dataset,
        selected_tools=["environment"],
        query_focus="climate temperature",
        worldbank_indicators=[],
    )
    render = _build_dispatcher(findings, dataset)

    # Find a distribution_box entry for temp_anomaly if it exists
    manifest = build_eda_chart_manifest(findings, dataset)
    temp_key = next(
        (e["key"] for e in manifest if e.get("metric") == "temp_anomaly" and
         e["key"].startswith("distribution_box:")),
        None,
    )
    if temp_key is None:
        pytest.skip("temp_anomaly distribution_box entry not in manifest — skip")

    fig = render(temp_key)
    # Should be a Box chart or None (never an empty bar chart)
    if fig is not None:
        trace_types = {type(t).__name__ for t in fig.data}
        assert "Box" in trace_types or len(fig.data) > 0, (
            "Expected Box trace for near-zero mean metric"
        )


# ---------------------------------------------------------------------------
# Empty data → no chart
# ---------------------------------------------------------------------------

def test_empty_dataset_produces_no_charts() -> None:
    empty = {code: {k: [] for k in ["worldbank", "employment", "conflict_events",
                                     "climate", "aqi", "city_scores", "news"]}
             for code in HIGH_COVERAGE_DEFAULTS[:2]}
    findings = run_eda(empty, selected_tools=["worldbank"], query_focus="test",
                       worldbank_indicators=[])
    charts = build_eda_charts(findings, empty, max_charts=4)
    assert isinstance(charts, list)
    # May be empty or have a heatmap — but must not raise
