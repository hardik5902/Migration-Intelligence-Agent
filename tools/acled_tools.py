"""ACLED conflict data (requires API key + email)."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import httpx

ACLED_READ = "https://api.acleddata.com/acled/read"


async def get_conflict_events(
    country_code: str,
    date_from: str,
    date_to: str,
    client: httpx.AsyncClient | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Fetch ACLED events for ISO3 country over a date range (YYYY-MM-DD)."""
    key = os.environ.get("ACLED_API_KEY", "")
    email = os.environ.get("ACLED_EMAIL", "")
    if not key or not email:
        return [], ACLED_READ
    params = {
        "key": key,
        "email": email,
        "iso": country_code.upper()[:3],
        "event_date": f"{date_from}|{date_to}",
        "event_date_where": "BETWEEN",
        "limit": 500,
    }
    own = client is None
    c = client or httpx.AsyncClient(timeout=90.0)
    rows: list[dict[str, Any]] = []
    built = ACLED_READ
    try:
        r = await c.get(ACLED_READ, params=params)
        r.raise_for_status()
        data = r.json()
        built = str(r.request.url)
        records = data.get("data", []) if isinstance(data, dict) else []
        for ev in records:
            if not isinstance(ev, dict):
                continue
            rows.append(
                {
                    "event_date": ev.get("event_date", ""),
                    "fatalities": float(ev.get("fatalities") or 0),
                    "event_type": ev.get("event_type", ""),
                    "country": ev.get("country", ""),
                    "latitude": ev.get("latitude"),
                    "longitude": ev.get("longitude"),
                    "source_api": "ACLED API",
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
