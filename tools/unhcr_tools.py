"""UNHCR population API (async httpx)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

UNHCR_BASE = "https://api.unhcr.org/population/v1/population"


async def get_displacement_data(
    country_code: str,
    year_from: int,
    year_to: int,
    client: httpx.AsyncClient | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Country of origin (coo) population statistics (refugees, asylum-seekers, etc.)."""
    code = country_code.upper()
    params = {
        "yearFrom": str(year_from),
        "yearTo": str(year_to),
        "coo": code,
        "limit": 500,
    }
    url = UNHCR_BASE
    own = client is None
    c = client or httpx.AsyncClient(timeout=60.0)
    built_url = url
    payload: Any = {}
    error_msg = None
    try:
        r = await c.get(url, params=params, follow_redirects=True)
        r.raise_for_status()
        payload = r.json()
        built_url = str(r.request.url)
    except Exception as e:
        error_msg = str(e)
        print(f"[UNHCR] Error fetching data for {code}: {e}")
        payload = {}
    finally:
        if own:
            await c.aclose()
    
    rows: list[dict[str, Any]] = []
    items = []
    if isinstance(payload, dict):
        items = payload.get("items") or payload.get("data") or []
    
    print(f"[UNHCR] Fetched {len(items)} items for {code} (coo={code}, {year_from}-{year_to})")
    
    for it in items:
        if not isinstance(it, dict):
            continue
        rows.append(
            {
                "country": code,
                "year": int(it.get("year") or it.get("time") or 0),
                "metric": it.get("populationType", {}).get("name")
                if isinstance(it.get("populationType"), dict)
                else str(it.get("populationType", "total")),
                "value": float(it.get("value") or 0),
                "coa": (it.get("countryOfAsylum") or {}).get("code", "")
                if isinstance(it.get("countryOfAsylum"), dict)
                else "",
                "coa_name": (it.get("countryOfAsylum") or {}).get("name", "")
                if isinstance(it.get("countryOfAsylum"), dict)
                else "",
                "source_api": "UNHCR API",
                "endpoint_url": built_url,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    
    if error_msg:
        print(f"[UNHCR] Error details: {error_msg}")
    if not rows:
        print(f"[UNHCR] No displacement data available for {code} in {year_from}-{year_to}")
    
    return rows, built_url
