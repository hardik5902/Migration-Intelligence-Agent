"""OpenAQ v2 latest measurements (public)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

OPENAQ_V2 = "https://api.openaq.org/v2/latest"


async def get_aqi_by_country(
    country_name_or_iso2: str,
    top_n: int = 30,
    client: httpx.AsyncClient | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Latest PM2.5 where available, keyed by location."""
    params: dict[str, Any] = {"limit": top_n, "parameter": "pm25", "order_by": "measurements"}
    q = country_name_or_iso2.strip().upper()
    if len(q) == 2:
        params["iso"] = q
    else:
        params["query"] = country_name_or_iso2
    own = client is None
    c = client or httpx.AsyncClient(timeout=45.0)
    rows: list[dict[str, Any]] = []
    built = OPENAQ_V2
    try:
        r = await c.get(OPENAQ_V2, params=params)
        r.raise_for_status()
        data = r.json()
        built = str(r.request.url)
        results = data.get("results", []) if isinstance(data, dict) else []
        for loc in results:
            if not isinstance(loc, dict):
                continue
            m = loc.get("measurements", [])
            pm = None
            for meas in m:
                if isinstance(meas, dict) and meas.get("parameter") == "pm25":
                    pm = float(meas.get("value") or 0)
                    break
            rows.append(
                {
                    "location": loc.get("location", ""),
                    "country": loc.get("country", ""),
                    "city": loc.get("city", ""),
                    "pm25": pm,
                    "coordinates": loc.get("coordinates"),
                    "source_api": "OpenAQ API",
                    "endpoint_url": built,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                }
            )
    except Exception:
        rows = []
    finally:
        if own:
            await c.aclose()
    return rows, built
