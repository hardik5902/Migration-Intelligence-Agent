"""World Bank API helpers (async httpx)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import httpx
from tools.country_codes import iso3_to_iso2

WB_BASE = "https://api.worldbank.org/v2/country"


async def get_indicator(
    country_code: str,
    indicator: str,
    year_from: int,
    year_to: int,
    client: httpx.AsyncClient | None = None,
) -> tuple[list[dict[str, Any]], str, str | None]:
    """Fetch a single World Bank indicator time series.

    Returns (rows, endpoint_url) where each row has year, value, indicator.
    """
    code = country_code.upper()
    params = {"format": "json", "date": f"{year_from}:{year_to}", "per_page": 500}
    url = f"{WB_BASE}/{code}/indicator/{indicator}"
    own = client is None
    c = client or httpx.AsyncClient(timeout=60.0)
    data = None
    error: str | None = None
    try:
        try:
            r = await c.get(url, params=params)
            # don't raise here — handle non-2xx gracefully and capture body
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError as exc:
                # capture response text for diagnostics
                try:
                    body = r.text
                except Exception:
                    body = str(exc)
                error = f"HTTP {r.status_code}: {body}"
                data = None
                # If World Bank rejects an ISO3 code with 400, try ISO2 fallback
                # (some endpoints accept alpha-2 in practice). Record any
                # successful fallback as the URL to return.
                if r.status_code == 400 and len(code) == 3:
                    iso2 = iso3_to_iso2(code)
                    if iso2:
                        try:
                            url2 = f"{WB_BASE}/{iso2}/indicator/{indicator}"
                            r2 = await c.get(url2, params=params)
                            r2.raise_for_status()
                            data = r2.json()
                            url = str(r2.url if hasattr(r2, "url") else url2)
                            error = None
                        except Exception as exc2:
                            # keep original error if fallback fails
                            try:
                                body2 = r2.text
                            except Exception:
                                body2 = str(exc2)
                            error = f"HTTP {r.status_code}: {body}; fallback-{iso2} error: {body2}"
                            data = None
            else:
                data = r.json()
                url = str(r.url if hasattr(r, "url") else url)
        except httpx.RequestError as exc:
            error = f"RequestError: {exc}"
            data = None
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
                    "endpoint_url": str(url),
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                }
            )
    # If date-range yielded no non-null values, try mrv (most recent values)
    if not rows:
        try:
            params_mrv = {"format": "json", "mrv": 15, "per_page": 500}
            r_mrv = await c.get(f"{WB_BASE}/{code}/indicator/{indicator}", params=params_mrv)
            r_mrv.raise_for_status()
            data_mrv = r_mrv.json()
            if isinstance(data_mrv, list) and len(data_mrv) > 1 and isinstance(data_mrv[1], list):
                for item in data_mrv[1]:
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
                            "endpoint_url": str(r_mrv.url if hasattr(r_mrv, "url") else url),
                            "fetched_at": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                url = str(r_mrv.url if hasattr(r_mrv, "url") else url)
        except Exception:
            pass

    return rows, url, error


async def fetch_macro_bundle(
    country_code: str,
    year_from: int,
    year_to: int,
    client: httpx.AsyncClient | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """GDP growth, inflation (CPI), stability, unemployment, Gini — all fetched in parallel."""
    indicators = {
        "NY.GDP.MKTP.KD.ZG": "gdp_growth",
        "FP.CPI.TOTL.ZG": "inflation",
        "PV.EST": "political_stability",
        "SL.UEM.TOTL.ZS": "unemployment",
        "SI.POV.GINI": "gini",
    }
    own = client is None
    c = client or httpx.AsyncClient(timeout=20.0)
    try:
        tasks = [
            get_indicator(country_code, ind, year_from, year_to, c)
            for ind in indicators
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        all_rows: list[dict[str, Any]] = []
        urls: list[str] = []
        for (ind, label), result in zip(indicators.items(), results):
            if isinstance(result, Exception):
                urls.append(f"ERROR:{ind}:{result}")
                continue
            rows, u, err = result
            for row in rows:
                row["label"] = label
            all_rows.extend(rows)
            urls.append(u)
            if err:
                urls.append(f"ERROR:{ind}:{err}")
    finally:
        if own:
            await c.aclose()
    return all_rows, urls


async def fetch_relocation_bundle(
    country_code: str,
    year_from: int,
    year_to: int,
    client: httpx.AsyncClient | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Country-level indicators used for relocation ranking — fetched in parallel."""
    indicators = {
        "SH.XPD.CHEX.GD.ZS": "health_expenditure_gdp",
        "SH.MED.PHYS.ZS": "physicians_per_1000",
        "SE.XPD.TOTL.GD.ZS": "education_spend_gdp",
        "NY.GDP.PCAP.CD": "gdp_per_capita_usd",
        "SI.POV.NAHC": "poverty_headcount",
    }
    own = client is None
    c = client or httpx.AsyncClient(timeout=20.0)
    try:
        tasks = [
            get_indicator(country_code, ind, year_from, year_to, c)
            for ind in indicators
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        all_rows: list[dict[str, Any]] = []
        urls: list[str] = []
        for (ind, label), result in zip(indicators.items(), results):
            if isinstance(result, Exception):
                urls.append(f"ERROR:{ind}:{result}")
                continue
            rows, url, err = result
            for row in rows:
                row["label"] = label
            all_rows.extend(rows)
            urls.append(url)
            if err:
                urls.append(f"ERROR:{ind}:{err}")
    finally:
        if own:
            await c.aclose()
    return all_rows, urls
