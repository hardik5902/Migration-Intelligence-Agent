"""Eval: Pydantic schema validation and UNHCR removal.

Verifies that schemas accept valid data, reject bad data gracefully,
and that the displacement field was removed from MigrationDataset.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from models.schemas import (
    MigrationDataset,
    ToolSelectorOutput,
    HypothesisInsight,
    ToolSelection,
)


# ---------------------------------------------------------------------------
# MigrationDataset
# ---------------------------------------------------------------------------

def test_migration_dataset_defaults() -> None:
    ds = MigrationDataset()
    assert ds.worldbank == []
    assert ds.employment == []
    assert ds.conflict_events == []


def test_migration_dataset_no_displacement_field() -> None:
    """displacement was removed when UNHCR was dropped."""
    ds = MigrationDataset()
    assert not hasattr(ds, "displacement"), (
        "MigrationDataset.displacement still exists — UNHCR removal incomplete"
    )


def test_migration_dataset_extra_fields_ignored() -> None:
    """extra='ignore' means passing unknown fields should not raise."""
    ds = MigrationDataset(displacement=[{"x": 1}])   # was a valid field before removal
    # Should silently ignore the unknown displacement kwarg (extra='ignore')
    assert isinstance(ds, MigrationDataset)


def test_migration_dataset_worldbank_accepts_list_of_dicts() -> None:
    ds = MigrationDataset(worldbank=[{"label": "gdp_growth", "value": 2.1, "year": 2020}])
    assert len(ds.worldbank) == 1


# ---------------------------------------------------------------------------
# ToolSelection
# ---------------------------------------------------------------------------

def test_tool_selection_no_unhcr_field() -> None:
    ts = ToolSelection()
    assert not hasattr(ts, "unhcr"), (
        "ToolSelection.unhcr still exists — UNHCR removal incomplete"
    )


def test_tool_selection_defaults() -> None:
    ts = ToolSelection()
    assert ts.worldbank is True
    assert ts.teleport is True


# ---------------------------------------------------------------------------
# ToolSelectorOutput
# ---------------------------------------------------------------------------

def test_tool_selector_output_defaults() -> None:
    out = ToolSelectorOutput()
    assert out.in_scope is True
    assert out.country_codes == []
    assert out.selected_tools == []


def test_tool_selector_output_roundtrip() -> None:
    raw = {
        "selected_tools": ["worldbank", "acled"],
        "countries": ["Germany", "Canada"],
        "country_codes": ["DEU", "CAN"],
        "country_strategy": "explicit_only",
        "k": 2,
        "query_focus": "conflict",
        "year_from": 2015,
        "year_to": 2023,
        "worldbank_indicators": ["political_stability"],
        "in_scope": True,
        "out_of_scope_reason": "",
        "reasoning": "test",
        "proxy_note": "",
    }
    out = ToolSelectorOutput.model_validate(raw)
    dumped = out.model_dump()
    assert dumped["country_codes"] == ["DEU", "CAN"]
    assert dumped["k"] == 2


def test_tool_selector_output_extra_fields_ignored() -> None:
    out = ToolSelectorOutput.model_validate({"unknown_key": "value"})
    assert isinstance(out, ToolSelectorOutput)


# ---------------------------------------------------------------------------
# HypothesisInsight
# ---------------------------------------------------------------------------

def test_hypothesis_insight_defaults() -> None:
    h = HypothesisInsight()
    assert h.confidence == 50
    assert h.evidence_for == []
    assert h.evidence_against == []


def test_hypothesis_insight_confidence_range() -> None:
    h = HypothesisInsight(confidence=75)
    assert 0 <= h.confidence <= 100


def test_hypothesis_insight_full_roundtrip() -> None:
    raw = {
        "headline": "Germany leads on healthcare",
        "summary": "Germany scores 8.5/10 on healthcare.",
        "key_metric": "Healthcare score 8.5, 2023",
        "reasoning": "High physician density and spending.",
        "evidence_for": ["physicians_per_1000: 4.3"],
        "evidence_against": ["cost_of_living is high"],
        "competing_hypothesis": "Netherlands may also qualify",
        "competing_verdict": "Germany edges out on spending",
        "data_source": "World Bank + Teleport",
        "confidence": 80,
    }
    h = HypothesisInsight.model_validate(raw)
    assert h.headline == raw["headline"]
    assert h.confidence == 80
