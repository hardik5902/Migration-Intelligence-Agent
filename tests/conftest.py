"""Shared pytest fixtures for the Migration Intelligence eval suite.

All fixtures are derived from live module constants — nothing is hardcoded.
"""

from __future__ import annotations

import pytest

from agents.pipeline_config import (
    AVAILABLE_TOOLS,
    HIGH_COVERAGE_DEFAULTS,
    TOOL_DATA_KEYS,
)
from tools.country_codes import all_country_names, country_name_to_iso3, iso3_to_name


# ---------------------------------------------------------------------------
# Minimal synthetic dataset factory
# Build realistic-shaped rows from the real schema, keyed on actual column
# names used by each tool's data_key. No real API calls needed.
# ---------------------------------------------------------------------------

def _wb_rows(iso3: str, labels: list[str] | None = None) -> list[dict]:
    """Minimal World Bank rows: one value per (label, year) pair."""
    from agents.eda_analyst import _TOOL_METRICS
    metric_labels = labels or _TOOL_METRICS.get("worldbank", [])[:4]
    rows = []
    base = 2015
    for label in metric_labels:
        for offset in range(6):
            rows.append({
                "country": iso3,
                "label": label,
                "value": float(offset + 1) * 1.5,
                "year": base + offset,
            })
    return rows


def _employment_rows(iso3: str) -> list[dict]:
    return [
        {"country": iso3, "unemployment_rate": 5.0 + i, "year": 2018 + i}
        for i in range(4)
    ]


def _conflict_rows(iso3: str) -> list[dict]:
    return [
        {"country": iso3, "event_type": "Battles", "fatalities": i * 10,
         "event_date": f"2020-01-{i+1:02d}", "year": 2020}
        for i in range(5)
    ]


def _climate_rows(iso3: str) -> list[dict]:
    return [
        {"country": iso3, "avg_temp_anomaly_c": 0.1 * i,
         "annual_precipitation_mm": 800 + i * 10, "year": 2018 + i}
        for i in range(5)
    ]


def _city_rows(iso3: str) -> list[dict]:
    return [
        {"country": iso3, "dimension": d, "score_out_of_10": 7.0 + j * 0.1}
        for j, d in enumerate(["Healthcare", "Education", "Safety", "Economy"])
    ]


def _news_rows(iso3: str) -> list[dict]:
    return [
        {"country": iso3, "title": f"Test headline {i}", "sentiment": 0.1 * i,
         "published_at": "2024-01-01"}
        for i in range(3)
    ]


@pytest.fixture
def available_tools() -> list[str]:
    return list(AVAILABLE_TOOLS.keys())


@pytest.fixture
def high_coverage_codes() -> list[str]:
    return list(HIGH_COVERAGE_DEFAULTS)


@pytest.fixture
def tool_data_keys() -> dict:
    return dict(TOOL_DATA_KEYS)


@pytest.fixture
def two_country_dataset() -> dict[str, dict]:
    """Minimal two-country dataset covering every active tool data key."""
    codes = HIGH_COVERAGE_DEFAULTS[:2]
    out = {}
    for code in codes:
        out[code] = {
            "worldbank": _wb_rows(code),
            "employment": _employment_rows(code),
            "conflict_events": _conflict_rows(code),
            "climate": _climate_rows(code),
            "aqi": [],
            "city_scores": _city_rows(code),
            "news": _news_rows(code),
        }
    return out


@pytest.fixture
def five_country_dataset() -> dict[str, dict]:
    """Five-country dataset using the high-coverage defaults."""
    codes = HIGH_COVERAGE_DEFAULTS
    out = {}
    for code in codes:
        out[code] = {
            "worldbank": _wb_rows(code),
            "employment": _employment_rows(code),
            "conflict_events": _conflict_rows(code),
            "climate": _climate_rows(code),
            "aqi": [],
            "city_scores": _city_rows(code),
            "news": _news_rows(code),
        }
    return out
