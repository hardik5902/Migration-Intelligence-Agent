"""Eval: country-code lookup correctness.

All test values are derived from the live _NAME_TO_ISO3 / iso3_to_iso2 maps
so the tests stay in sync automatically when new countries are added.
"""

from __future__ import annotations

import pytest

from tools.country_codes import (
    all_country_names,
    country_name_to_iso3,
    iso3_to_iso2,
    iso3_to_name,
    normalize_name,
)
from agents.pipeline_config import HIGH_COVERAGE_DEFAULTS


# ---------------------------------------------------------------------------
# Basic round-trips for every registered country name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", all_country_names())
def test_every_registered_name_resolves_to_iso3(name: str) -> None:
    code = country_name_to_iso3(name)
    assert code and len(code) == 3, f"{name!r} → {code!r} (expected 3-char ISO3)"


@pytest.mark.parametrize("name", all_country_names())
def test_every_iso3_round_trips_back_to_name(name: str) -> None:
    code = country_name_to_iso3(name)
    assert code
    back = iso3_to_name(code)
    assert back, f"iso3_to_name({code!r}) returned nothing (via {name!r})"


# ---------------------------------------------------------------------------
# High-coverage defaults must all resolve
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("code", HIGH_COVERAGE_DEFAULTS)
def test_high_coverage_code_has_name(code: str) -> None:
    assert iso3_to_name(code), f"{code} has no human-readable name"


@pytest.mark.parametrize("code", HIGH_COVERAGE_DEFAULTS)
def test_high_coverage_code_has_iso2(code: str) -> None:
    assert iso3_to_iso2(code), f"{code} has no ISO2 mapping (needed for some APIs)"


# ---------------------------------------------------------------------------
# normalize_name: spacing and casing should not break lookups
# ---------------------------------------------------------------------------

def test_normalize_strips_whitespace() -> None:
    assert normalize_name("  Germany  ") == "germany"


def test_normalize_lowercases() -> None:
    assert normalize_name("SOUTH AFRICA") == "south africa"


def test_lookup_case_insensitive() -> None:
    assert country_name_to_iso3("GERMANY") == country_name_to_iso3("germany")


# ---------------------------------------------------------------------------
# Specific countries that historically caused bugs (explicit regression tests)
# ---------------------------------------------------------------------------

REGRESSION_COUNTRIES = [
    ("south africa", "ZAF"),
    ("germany", "DEU"),
    ("canada", "CAN"),
    ("australia", "AUS"),
    ("netherlands", "NLD"),
    ("sweden", "SWE"),
    ("india", "IND"),
    ("brazil", "BRA"),
    ("japan", "JPN"),
]


@pytest.mark.parametrize("name,expected_code", REGRESSION_COUNTRIES)
def test_regression_country_lookup(name: str, expected_code: str) -> None:
    assert country_name_to_iso3(name) == expected_code


# ---------------------------------------------------------------------------
# Unknown names return None gracefully
# ---------------------------------------------------------------------------

def test_unknown_name_returns_none() -> None:
    assert country_name_to_iso3("Narnia") is None


def test_unknown_iso3_returns_none() -> None:
    assert iso3_to_name("ZZZ") is None
