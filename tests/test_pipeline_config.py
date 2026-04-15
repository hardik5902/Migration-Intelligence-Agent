"""Eval: pipeline configuration consistency.

Checks that AVAILABLE_TOOLS, TOOL_DATA_KEYS, and TOOL_DEFAULT_COUNTRY_POOLS
are internally consistent — every tool the pipeline can select must have a
data-key mapping, and every pool code must be resolvable.
"""

from __future__ import annotations

import pytest

from agents.pipeline_config import (
    AVAILABLE_TOOLS,
    HIGH_COVERAGE_DEFAULTS,
    TOOL_DATA_KEYS,
    TOOL_DEFAULT_COUNTRY_POOLS,
)
from tools.country_codes import iso3_to_name


# ---------------------------------------------------------------------------
# AVAILABLE_TOOLS integrity
# ---------------------------------------------------------------------------

def test_available_tools_is_non_empty() -> None:
    assert AVAILABLE_TOOLS, "AVAILABLE_TOOLS must not be empty"


def test_every_available_tool_has_data_key_mapping() -> None:
    missing = [t for t in AVAILABLE_TOOLS if t not in TOOL_DATA_KEYS]
    assert not missing, f"Tools missing TOOL_DATA_KEYS entry: {missing}"


def test_every_data_key_entry_is_non_empty_list() -> None:
    for tool, keys in TOOL_DATA_KEYS.items():
        assert isinstance(keys, list) and keys, (
            f"TOOL_DATA_KEYS[{tool!r}] must be a non-empty list, got {keys!r}"
        )


# ---------------------------------------------------------------------------
# HIGH_COVERAGE_DEFAULTS integrity
# ---------------------------------------------------------------------------

def test_high_coverage_defaults_are_valid_iso3() -> None:
    for code in HIGH_COVERAGE_DEFAULTS:
        assert len(code) == 3, f"{code!r} is not 3-char ISO3"
        assert iso3_to_name(code), f"{code!r} has no resolvable name"


def test_high_coverage_defaults_has_minimum_count() -> None:
    assert len(HIGH_COVERAGE_DEFAULTS) >= 3, (
        "Need at least 3 high-coverage defaults for meaningful comparison"
    )


# ---------------------------------------------------------------------------
# TOOL_DEFAULT_COUNTRY_POOLS integrity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tool", list(TOOL_DEFAULT_COUNTRY_POOLS.keys()))
def test_pool_tool_is_registered(tool: str) -> None:
    assert tool in AVAILABLE_TOOLS, (
        f"TOOL_DEFAULT_COUNTRY_POOLS has pool for unregistered tool {tool!r}"
    )


@pytest.mark.parametrize("tool", list(TOOL_DEFAULT_COUNTRY_POOLS.keys()))
def test_pool_codes_are_resolvable(tool: str) -> None:
    for code in TOOL_DEFAULT_COUNTRY_POOLS[tool]:
        assert iso3_to_name(code), (
            f"Pool code {code!r} for tool {tool!r} cannot be resolved to a country name"
        )


# ---------------------------------------------------------------------------
# No UNHCR / displacement references remain in active config
# ---------------------------------------------------------------------------

def test_unhcr_not_in_available_tools() -> None:
    assert "unhcr" not in AVAILABLE_TOOLS


def test_unhcr_not_in_tool_data_keys() -> None:
    assert "unhcr" not in TOOL_DATA_KEYS


def test_displacement_not_in_any_data_key_list() -> None:
    for tool, keys in TOOL_DATA_KEYS.items():
        assert "displacement" not in keys, (
            f"displacement key still present under TOOL_DATA_KEYS[{tool!r}]"
        )
