"""Employment / labor market — World Bank unemployment as reliable primary source."""

from __future__ import annotations

from typing import Any

import httpx

from tools.worldbank_tools import get_indicator


async def get_employment_data(
    country_code: str,
    year_from: int,
    year_to: int,
    client: httpx.AsyncClient | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Unemployment (% total labor force) and labor participation proxy via WB series."""
    rows, url = await get_indicator(
        country_code, "SL.UEM.TOTL.ZS", year_from, year_to, client
    )
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "country": r["country"],
                "year": r["year"],
                "unemployment_rate": r["value"],
                "youth_unemployment_rate": None,
                "labor_force_participation": None,
                "source_api": "ILO via World Bank proxy (SL.UEM.TOTL.ZS)",
                "indicator_code": "SL.UEM.TOTL.ZS",
                "endpoint_url": url,
                "fetched_at": r["fetched_at"],
            }
        )
    return out, url
