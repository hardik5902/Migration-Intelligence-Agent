"""Step 2 of the country pipeline: fetch live data for every selected country
in parallel using the scout service.
"""

from __future__ import annotations

import asyncio
from typing import Any

from agents.pipeline_config import AVAILABLE_TOOLS
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
    tool_flags = {t: (t in selected_tools) for t in AVAILABLE_TOOLS}

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
