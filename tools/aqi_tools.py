"""OpenAQ latest measurements with optional API key."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import httpx

OPENAQ_V3 = "https://api.openaq.org/v3/locations"
OPENAQ_V2 = "https://api.openaq.org/v2/latest"


def _headers() -> dict[str, str]:
    key = os.environ.get("OPENAQ_API_KEY", "").strip()
    return {"X-API-Key": key} if key else {}


async def _fetch_v3(
    country_name_or_iso2: str,
    top_n: int,
    client: httpx.AsyncClient,
) -> tuple[list[dict[str, Any]], str]:
    params: dict[str, Any] = {"limit": top_n, "parameter": "pm25"}
    q = country_name_or_iso2.strip().upper()
    if len(q) == 2:
        params["country"] = q
    else:
        params["query"] = country_name_or_iso2
    response = await client.get(OPENAQ_V3, params=params, headers=_headers())
    response.raise_for_status()
    payload = response.json()
    built = str(response.request.url)
    results = payload.get("results", []) if isinstance(payload, dict) else []
    rows: list[dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        country = ""
        if isinstance(item.get("country"), dict):
            country = item["country"].get("code", "")
        measurements = item.get("sensors", []) or item.get("parameters", []) or []
        pm25 = None
        for measurement in measurements:
            if not isinstance(measurement, dict):
                continue
            param = measurement.get("parameter") or measurement.get("name")
            if str(param).lower() != "pm25":
                continue
            summary = measurement.get("summary") or {}
            pm25 = summary.get("avg") if isinstance(summary, dict) else None
            if pm25 is None:
                pm25 = measurement.get("lastValue") or measurement.get("value")
            break
        rows.append(
            {
                "location": item.get("name", ""),
                "country": country,
                "city": item.get("city", ""),
                "pm25": float(pm25) if pm25 is not None else None,
                "coordinates": item.get("coordinates"),
                "source_api": "OpenAQ API",
                "endpoint_url": built,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    return rows, built


async def _fetch_v2(
    country_name_or_iso2: str,
    top_n: int,
    client: httpx.AsyncClient,
) -> tuple[list[dict[str, Any]], str]:
    params: dict[str, Any] = {"limit": top_n, "parameter": "pm25", "order_by": "measurements"}
    q = country_name_or_iso2.strip().upper()
    if len(q) == 2:
        params["country"] = q
    else:
        params["city"] = country_name_or_iso2
    response = await client.get(OPENAQ_V2, params=params, headers=_headers())
    response.raise_for_status()
    payload = response.json()
    built = str(response.request.url)
    results = payload.get("results", []) if isinstance(payload, dict) else []
    rows: list[dict[str, Any]] = []
    for loc in results:
        if not isinstance(loc, dict):
            continue
        pm25 = None
        for measurement in loc.get("measurements", []) or []:
            if isinstance(measurement, dict) and measurement.get("parameter") == "pm25":
                pm25 = measurement.get("value")
                break
        rows.append(
            {
                "location": loc.get("location", ""),
                "country": loc.get("country", ""),
                "city": loc.get("city", ""),
                "pm25": float(pm25) if pm25 is not None else None,
                "coordinates": loc.get("coordinates"),
                "source_api": "OpenAQ API",
                "endpoint_url": built,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    return rows, built


async def get_aqi_by_country(
    country_name_or_iso2: str,
    top_n: int = 30,
    client: httpx.AsyncClient | None = None,
) -> tuple[list[dict[str, Any]], str]:
    own = client is None
    c = client or httpx.AsyncClient(timeout=45.0)
    try:
        try:
            rows, built = await _fetch_v3(country_name_or_iso2, top_n, c)
        except Exception:
            rows, built = await _fetch_v2(country_name_or_iso2, top_n, c)
        rows = [row for row in rows if row.get("pm25") is not None]
        return rows, built
    except Exception:
        return [], OPENAQ_V3
    finally:
        if own:
            await c.aclose()
