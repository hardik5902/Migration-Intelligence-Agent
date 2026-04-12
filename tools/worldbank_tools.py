"""World Bank API helpers (async httpx)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

WB_BASE = "https://api.worldbank.org/v2/country"


async def get_indicator(
    country_code: str,
    indicator: str,
    year_from: int,
    year_to: int,
    client: httpx.AsyncClient | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Fetch a single World Bank indicator time series.

    Returns (rows, endpoint_url) where each row has year, value, indicator.
    """
    code = country_code.upper()
    url = (
        f"{WB_BASE}/{code}/indicator/{indicator}"
        f"?date={year_from}:{year_to}&format=json&per_page=500"
    )
    own = client is None
    c = client or httpx.AsyncClient(timeout=60.0)
    try:
        r = await c.get(url)
        r.raise_for_status()
        data = r.json()
    finally:
        if own:
            await c.aclose()
    rows: list[dict[str, Any]] = []
    if isinstance(data, list) and len(data) > 1 and isinstance(data[1], list):
        for item in data[1]:
            if not isinstance(item, dict):
                continue
            val = item.get("value")
            if val is None:
                continue
            rows.append(
                {
                    "country": code,
                    "year": int(item.get("date", 0) or 0),
                    "indicator": indicator,
                    "value": float(val),
                    "source_api": "World Bank API",
                    "endpoint_url": url,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                }
            )
    return rows, url


async def fetch_macro_bundle(
    country_code: str,
    year_from: int,
    year_to: int,
    client: httpx.AsyncClient | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """GDP growth, inflation (CPI), stability, unemployment, Gini in one bundle."""
    indicators = {
        "NY.GDP.MKTP.KD.ZG": "gdp_growth",
        "FP.CPI.TOTL.ZG": "inflation",
        "PV.EST": "political_stability",
        "SL.UEM.TOTL.ZS": "unemployment",
        "SI.POV.GINI": "gini",
    }
    own = client is None
    c = client or httpx.AsyncClient(timeout=60.0)
    all_rows: list[dict[str, Any]] = []
    urls: list[str] = []
    try:
        for ind, _label in indicators.items():
            rows, u = await get_indicator(code, ind, year_from, year_to, c)
            for row in rows:
                row["label"] = indicators[ind]
            all_rows.extend(rows)
            urls.append(u)
    finally:
        if own:
            await c.aclose()
    return all_rows, urls
