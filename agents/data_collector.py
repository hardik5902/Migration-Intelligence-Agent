"""Step 2 of the country pipeline: fetch live data for every selected country
in parallel using the scout service.
"""

from __future__ import annotations

import asyncio
from typing import Any

from agents.pipeline_config import AVAILABLE_TOOLS, TOOL_DATA_KEYS
from agents.scout_service import collect_migration_dataset
from models.schemas import IntentConfig


async def collect_data_for_countries(
    countries: list[str],
    country_codes: list[str],
    selected_tools: list[str],
    year_from: int,
    year_to: int,
) -> dict[str, Any]:
    """Fetch data for every country in parallel.

    Returns a dict mapping country name → dataset dict (from MigrationDataset.model_dump()).
    """
    # Build per-tool flags: selected tools enabled; teleport always on (free WB proxy).
    tool_flags = {t: (t in selected_tools) for t in AVAILABLE_TOOLS}
    tool_flags["teleport"] = True  # always collect quality-of-life composite scores

    async def fetch_one(country: str, code: str) -> tuple[str, Any]:
        intent = IntentConfig(
            intent="push_factor",
            country=country,
            country_code=code,
            year_from=year_from,
            year_to=year_to,
        )
        try:
            dataset = await collect_migration_dataset(intent, tool_flags)
            return country, dataset.model_dump()
        except Exception as exc:
            print(f"[DATA_COLLECTOR] Failed for {country}: {exc}")
            return country, {
                "error": str(exc),
                "country": country,
                "worldbank": [],
                "employment": [],
                "aqi": [],
                "climate": [],
                "displacement": [],
                "conflict_events": [],
                "city_scores": [],
                "news": [],
            }

    tasks = [fetch_one(c, code) for c, code in zip(countries, country_codes)]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    out: dict[str, Any] = {}
    for item in raw_results:
        if isinstance(item, Exception):
            print(f"[DATA_COLLECTOR] gather exception: {item}")
            continue
        country, data = item
        out[country] = data

    return out


def rank_countries_by_coverage(
    countries_data: dict[str, Any],
    selected_tools: list[str],
) -> list[dict[str, Any]]:
    """Rank countries by how much useful data actually came back."""
    ranked: list[dict[str, Any]] = []

    for country, dataset in countries_data.items():
        matched_tools = 0
        total_rows = 0
        per_tool: dict[str, int] = {}

        for tool in selected_tools:
            tool_rows = sum(
                len(dataset.get(key) or [])
                for key in TOOL_DATA_KEYS.get(tool, [tool])
            )
            per_tool[tool] = tool_rows
            total_rows += tool_rows
            if tool_rows > 0:
                matched_tools += 1

        ranked.append(
            {
                "country": country,
                "matched_tools": matched_tools,
                "total_rows": total_rows,
                "per_tool": per_tool,
            }
        )

    ranked.sort(
        key=lambda item: (
            item["matched_tools"],
            item["total_rows"],
        ),
        reverse=True,
    )
    return ranked


def filter_countries_by_coverage(
    countries_data: dict[str, Any],
    country_codes: list[str],
    selected_tools: list[str],
    k: int,
) -> tuple[dict[str, Any], list[str], list[str], list[dict[str, Any]]]:
    """Keep the best-supported countries before EDA runs."""
    ranked = rank_countries_by_coverage(countries_data, selected_tools)
    if not ranked:
        return countries_data, list(countries_data.keys()), country_codes, []

    code_by_country = {
        country: code
        for country, code in zip(countries_data.keys(), country_codes)
    }

    keep = [item for item in ranked if item["matched_tools"] > 0]
    if len(keep) < min(2, len(ranked)):
        keep = ranked[: min(max(k, 2), len(ranked))]
    else:
        keep = keep[: min(max(k, 2), len(keep))]

    kept_countries = [item["country"] for item in keep]
    kept_codes = [code_by_country.get(country, "") for country in kept_countries]
    filtered_data = {
        country: countries_data[country]
        for country in kept_countries
        if country in countries_data
    }

    return filtered_data, kept_countries, kept_codes, ranked
