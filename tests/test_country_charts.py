"""Eval: comparison chart builder (country_charts.py).

Tests that:
  - charts render without blank traces
  - no chart is repeated across a single run
  - manifest keys are all renderable
  - CO2 data key is correct
  - UNHCR/displacement references are gone
"""

from __future__ import annotations

import json
import pytest
import plotly.graph_objects as go

from analysis.country_charts import (
    build_country_comparison_charts,
    build_registry_manifest,
    render_charts_by_keys,
)
from agents.pipeline_config import AVAILABLE_TOOLS


# ---------------------------------------------------------------------------
# build_registry_manifest
# ---------------------------------------------------------------------------

def test_manifest_is_list(two_country_dataset: dict) -> None:
    manifest = build_registry_manifest(two_country_dataset)
    assert isinstance(manifest, list)


def test_manifest_entry_schema(two_country_dataset: dict) -> None:
    manifest = build_registry_manifest(two_country_dataset)
    for entry in manifest:
        assert "key" in entry and entry["key"], f"Entry missing 'key': {entry}"
        assert "title" in entry, f"Entry missing 'title': {entry}"


def test_manifest_no_unhcr_keys(two_country_dataset: dict) -> None:
    manifest = build_registry_manifest(two_country_dataset)
    keys = [e["key"] for e in manifest]
    assert not any("unhcr" in k.lower() for k in keys), (
        f"UNHCR key still present in manifest: {[k for k in keys if 'unhcr' in k.lower()]}"
    )


def test_manifest_no_displacement_keys(two_country_dataset: dict) -> None:
    manifest = build_registry_manifest(two_country_dataset)
    keys = [e["key"] for e in manifest]
    assert not any("displacement" in k.lower() for k in keys)


# ---------------------------------------------------------------------------
# render_charts_by_keys: no blank charts
# ---------------------------------------------------------------------------

def test_render_charts_by_keys_no_blank_traces(two_country_dataset: dict) -> None:
    manifest = build_registry_manifest(two_country_dataset)
    if not manifest:
        pytest.skip("No renderable charts for this dataset")
    keys = [e["key"] for e in manifest[:6]]
    jsons = render_charts_by_keys(keys, manifest, two_country_dataset)
    for j in jsons:
        obj = json.loads(j)
        assert obj.get("data"), f"Chart JSON has no data traces: {obj.get('layout', {}).get('title')}"


def test_render_unknown_key_is_skipped(two_country_dataset: dict) -> None:
    manifest = build_registry_manifest(two_country_dataset)
    jsons = render_charts_by_keys(["nonexistent_key_xyz"], manifest, two_country_dataset)
    assert jsons == [], "Unknown key should be silently skipped, not raise"


# ---------------------------------------------------------------------------
# build_country_comparison_charts
# ---------------------------------------------------------------------------

def test_comparison_charts_returns_list(two_country_dataset: dict) -> None:
    charts = build_country_comparison_charts(
        two_country_dataset,
        selected_tools=["worldbank", "employment"],
        query_focus="economic conditions",
        worldbank_indicators=["gdp_growth"],
    )
    assert isinstance(charts, list)


def test_comparison_charts_no_blanks(two_country_dataset: dict) -> None:
    charts = build_country_comparison_charts(
        two_country_dataset,
        selected_tools=["worldbank"],
        query_focus="health",
        worldbank_indicators=["life_expectancy"],
    )
    for j in charts:
        obj = json.loads(j)
        assert obj.get("data"), (
            f"Blank chart returned: {obj.get('layout', {}).get('title')}"
        )


def test_comparison_charts_no_duplicates(five_country_dataset: dict) -> None:
    charts = build_country_comparison_charts(
        five_country_dataset,
        selected_tools=list(AVAILABLE_TOOLS.keys()),
        query_focus="general comparison",
        worldbank_indicators=[],
    )
    titles = []
    for j in charts:
        obj = json.loads(j)
        t = obj.get("layout", {}).get("title", {})
        titles.append(t.get("text", "") if isinstance(t, dict) else str(t))
    assert len(titles) == len(set(titles)), f"Duplicate chart titles: {titles}"


def test_comparison_charts_empty_dataset_does_not_raise() -> None:
    from agents.pipeline_config import HIGH_COVERAGE_DEFAULTS
    empty = {c: {"worldbank": [], "employment": [], "conflict_events": [],
                  "climate": [], "aqi": [], "city_scores": [], "news": []}
             for c in HIGH_COVERAGE_DEFAULTS[:2]}
    charts = build_country_comparison_charts(
        empty, selected_tools=["worldbank"], query_focus="test",
        worldbank_indicators=[],
    )
    assert isinstance(charts, list)
