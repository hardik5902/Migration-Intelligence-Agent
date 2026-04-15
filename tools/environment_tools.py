"""Combined environment tool: climate (Open-Meteo) + air quality (OpenAQ) in parallel."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from tools import aqi_tools, climate_tools


async def get_environment_data(
    country_code: str,
    year_from: int,
    year_to: int,
    client: httpx.AsyncClient | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch climate trends and air quality data in parallel.

    Returns (climate_rows, aqi_rows).
    - climate_rows: annual temperature anomaly, precipitation, heat days (Open-Meteo)
    - aqi_rows:     PM2.5 measurements by city (OpenAQ)
    """
    own = client is None
    c = client or httpx.AsyncClient(timeout=20.0)
    try:
        climate_res, aqi_res = await asyncio.gather(
            climate_tools.get_climate_data(country_code, year_from, year_to, c),
            aqi_tools.get_aqi_by_country(country_code, top_n=40, client=c),
            return_exceptions=True,
        )

        # Unpack climate result (returns tuple or raises)
        climate_rows: list[dict[str, Any]] = []
        if not isinstance(climate_res, Exception):
            if isinstance(climate_res, tuple) and climate_res:
                climate_rows = climate_res[0] or []
            elif isinstance(climate_res, list):
                climate_rows = climate_res

        # Unpack AQI result
        aqi_rows: list[dict[str, Any]] = []
        if not isinstance(aqi_res, Exception):
            if isinstance(aqi_res, tuple) and aqi_res:
                aqi_rows = aqi_res[0] or []
            elif isinstance(aqi_res, list):
                aqi_rows = aqi_res

        return climate_rows, aqi_rows
    finally:
        if own:
            await c.aclose()
