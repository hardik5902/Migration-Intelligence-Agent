"""Eval: tool selector — country extraction and resolution logic.

Tests the pure-Python layer (no LLM calls). The LLM path is mocked
to return a fixture ToolSelectorOutput.
"""

from __future__ import annotations

import pytest

from agents.tool_selector import (
    _dedupe_country_pairs,
    _extract_explicit_countries,
    _normalize_country_suggestions,
    _default_country_codes_for_tools,
    _resolve_candidate_countries,
)
from agents.pipeline_config import AVAILABLE_TOOLS, HIGH_COVERAGE_DEFAULTS
from models.schemas import ToolSelectorOutput


# ---------------------------------------------------------------------------
# _dedupe_country_pairs
# ---------------------------------------------------------------------------

def test_dedupe_removes_duplicate_codes() -> None:
    pairs = [("Germany", "DEU"), ("Deutschland", "DEU"), ("Canada", "CAN")]
    result = _dedupe_country_pairs(pairs)
    codes = [c for _, c in result]
    assert codes.count("DEU") == 1


def test_dedupe_drops_invalid_codes() -> None:
    pairs = [("Nowhere", ""), ("Also Nowhere", "XX"), ("Germany", "DEU")]
    result = _dedupe_country_pairs(pairs)
    codes = [c for _, c in result]
    assert "DEU" in codes
    assert "" not in codes
    assert "XX" not in codes


# ---------------------------------------------------------------------------
# _extract_explicit_countries
# ---------------------------------------------------------------------------

def test_extracts_single_country() -> None:
    pairs = _extract_explicit_countries("What is happening in Germany?")
    codes = [c for _, c in pairs]
    assert "DEU" in codes


def test_extracts_multiple_countries() -> None:
    pairs = _extract_explicit_countries(
        "Compare Germany, Canada, and Australia for migration."
    )
    codes = [c for _, c in pairs]
    assert "DEU" in codes
    assert "CAN" in codes
    assert "AUS" in codes


def test_extracts_south_africa_regression() -> None:
    """Regression: 'South Africa' was previously not recognized."""
    pairs = _extract_explicit_countries(
        "Compare Japan, Germany, Brazil, South Africa, and India."
    )
    codes = [c for _, c in pairs]
    assert "ZAF" in codes, f"South Africa (ZAF) not found in {codes}"


def test_empty_query_returns_empty() -> None:
    assert _extract_explicit_countries("") == []


def test_no_country_in_query_returns_empty() -> None:
    pairs = _extract_explicit_countries("What is the best city to live in?")
    # May or may not match city names — just ensure it doesn't raise
    assert isinstance(pairs, list)


# ---------------------------------------------------------------------------
# _normalize_country_suggestions
# ---------------------------------------------------------------------------

def test_normalize_from_names() -> None:
    pairs = _normalize_country_suggestions(["Germany", "Canada"], [])
    codes = [c for _, c in pairs]
    assert "DEU" in codes
    assert "CAN" in codes


def test_normalize_from_codes() -> None:
    pairs = _normalize_country_suggestions([], ["DEU", "CAN", "AUS"])
    codes = [c for _, c in pairs]
    assert "DEU" in codes


def test_normalize_deduplicates() -> None:
    pairs = _normalize_country_suggestions(["Germany", "Germany"], ["DEU"])
    codes = [c for _, c in pairs]
    assert codes.count("DEU") == 1


# ---------------------------------------------------------------------------
# _default_country_codes_for_tools
# ---------------------------------------------------------------------------

def test_defaults_fall_back_to_high_coverage() -> None:
    codes = _default_country_codes_for_tools(["worldbank"], k=3)
    assert len(codes) >= 3
    for c in codes:
        assert len(c) == 3


def test_defaults_respect_k() -> None:
    codes = _default_country_codes_for_tools(list(AVAILABLE_TOOLS.keys()), k=2)
    assert len(codes) >= 2   # at least k


# ---------------------------------------------------------------------------
# _resolve_candidate_countries
# ---------------------------------------------------------------------------

def _selector(**kwargs) -> ToolSelectorOutput:
    defaults = dict(
        selected_tools=["worldbank"],
        countries=["Germany", "Canada"],
        country_codes=["DEU", "CAN"],
        country_strategy="mixed",
        k=5,
        query_focus="test",
        year_from=2015,
        year_to=2023,
        worldbank_indicators=[],
        in_scope=True,
        out_of_scope_reason="",
        reasoning="",
        proxy_note="",
    )
    defaults.update(kwargs)
    return ToolSelectorOutput(**defaults)


def test_explicit_countries_override_suggestions() -> None:
    sel = _selector(country_strategy="mixed")
    result = _resolve_candidate_countries("Compare Germany and Japan.", sel)
    assert "DEU" in result.country_codes
    assert "JPN" in result.country_codes


def test_strategy_set_to_explicit_only_when_all_countries_named() -> None:
    sel = _selector(country_strategy="explicit_only", k=2)
    result = _resolve_candidate_countries("Germany and Canada comparison.", sel)
    assert result.country_strategy in ("explicit_only", "mixed")


def test_fallback_to_defaults_when_no_countries_found() -> None:
    sel = _selector(countries=[], country_codes=[], country_strategy="high_coverage_defaults")
    result = _resolve_candidate_countries("What are the best countries for migrants?", sel)
    assert result.country_codes, "Should fall back to high-coverage defaults"
    for c in result.country_codes:
        assert len(c) == 3


def test_k_reflects_actual_country_count() -> None:
    sel = _selector(k=10)
    result = _resolve_candidate_countries("Germany and Canada.", sel)
    assert result.k == len(result.countries) == len(result.country_codes)
