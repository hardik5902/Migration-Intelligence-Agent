"""Employment / labor market from ILOSTAT with a World Bank fallback."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import httpx

from tools.worldbank_tools import get_indicator

ILOSTAT_BASE = "https://sdmx.ilo.org/rest/data/ILO,DF_STI_ALL_UNE_TUNE_SEX_AGE_RT"


def _parse_ilo_obs(payload: Any, country_code: str) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data") or payload.get("dataSets") or payload.get("dataSet") or []
    structure = payload.get("structure") or {}
    if not isinstance(data, list) or not data:
        return []
    observations = data[0].get("observations") if isinstance(data[0], dict) else None
    if not isinstance(observations, dict):
        return []

    rows: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()
    for key, value in observations.items():
        parts = str(key).split(":")
        year = None
        for part in reversed(parts):
            if part.isdigit() and len(part) == 4:
                year = int(part)
                break
        if year is None:
            continue
        obs_value = None
        if isinstance(value, list) and value:
            obs_value = value[0]
        elif isinstance(value, dict):
            obs_value = value.get("value")
        if obs_value is None:
            continue
        rows.append(
            {
                "country": country_code,
                "year": year,
                "unemployment_rate": float(obs_value),
                "youth_unemployment_rate": None,
                "labor_force_participation": None,
                "source_api": "ILO ILOSTAT API",
                "indicator_code": "UNE_TUNE_RT_A",
                "endpoint_url": ILOSTAT_BASE,
                "fetched_at": now,
            }
        )
    return sorted(rows, key=lambda row: row["year"])


async def _try_ilostat(
    country_code: str,
    year_from: int,
    year_to: int,
    client: httpx.AsyncClient,
) -> tuple[list[dict[str, Any]], str]:
    code = country_code.upper()
    candidates = [
        (
            f"{ILOSTAT_BASE}/A.{code}.SEX_T.AGE_AGGREGATE",
            {"startPeriod": str(year_from), "endPeriod": str(year_to), "format": "sdmx-json"},
        ),
        (
            f"{ILOSTAT_BASE}/A.{code}..",
            {"startPeriod": str(year_from), "endPeriod": str(year_to), "format": "sdmx-json"},
        ),
    ]
    for base_url, params in candidates:
        try:
            response = await client.get(base_url, params=params)
            response.raise_for_status()
            rows = _parse_ilo_obs(response.json(), code)
            if rows:
                built = str(response.request.url)
                for row in rows:
                    row["endpoint_url"] = built
                return rows, built
        except Exception:
            continue
    return [], ILOSTAT_BASE


async def get_employment_data(
    country_code: str,
    year_from: int,
    year_to: int,
    client: httpx.AsyncClient | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Labor-market series: ILOSTAT first, then 3 World Bank indicators in parallel."""
    own = client is None
    c = client or httpx.AsyncClient(timeout=20.0)
    try:
        rows, url = await _try_ilostat(country_code, year_from, year_to, c)
        if rows:
            return rows, url

        # Fetch all three WB indicators concurrently instead of sequentially
        (unemployment_rows, url, _), (youth_rows, _, _), (participation_rows, _, _) = (
            await asyncio.gather(
                get_indicator(country_code, "SL.UEM.TOTL.ZS", year_from, year_to, c),
                get_indicator(country_code, "SL.UEM.1524.ZS", year_from, year_to, c),
                get_indicator(country_code, "SL.TLF.CACT.ZS", year_from, year_to, c),
                return_exceptions=False,
            )
        )

        youth_by_year = {row["year"]: row["value"] for row in youth_rows}
        participation_by_year = {row["year"]: row["value"] for row in participation_rows}
        out: list[dict[str, Any]] = []
        for row in unemployment_rows:
            year = row["year"]
            out.append(
                {
                    "country": row["country"],
                    "year": year,
                    "unemployment_rate": row["value"],
                    "youth_unemployment_rate": youth_by_year.get(year),
                    "labor_force_participation": participation_by_year.get(year),
                    "source_api": "World Bank labor proxy",
                    "indicator_code": "SL.UEM.TOTL.ZS",
                    "endpoint_url": url,
                    "fetched_at": row["fetched_at"],
                }
            )
        return out, url
    finally:
        if own:
            await c.aclose()
