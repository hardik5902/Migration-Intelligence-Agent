"""ACLED conflict data (OAuth authentication)."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import httpx

ACLED_OAUTH_TOKEN = "https://acleddata.com/oauth/token"
ACLED_READ = "https://acleddata.com/api/acled/read"


async def _get_acled_token(client: httpx.AsyncClient) -> str | None:
    """Get OAuth Bearer token using username/password."""
    username = os.environ.get("ACLED_USERNAME", "")
    password = os.environ.get("ACLED_PASSWORD", "")
    if not username or not password:
        return None
    try:
        data = {
            "username": username,
            "password": password,
            "grant_type": "password",
            "client_id": "acled",
        }
        r = await client.post(ACLED_OAUTH_TOKEN, data=data)
        r.raise_for_status()
        token_data = r.json()
        return token_data.get("access_token")
    except Exception:
        return None


async def get_conflict_events(
    country_code: str,
    date_from: str,
    date_to: str,
    client: httpx.AsyncClient | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Fetch ACLED events for ISO3 country over a date range (YYYY-MM-DD)."""
    own = client is None
    c = client or httpx.AsyncClient(timeout=90.0)
    rows: list[dict[str, Any]] = []
    built = ACLED_READ
    try:
        # Get OAuth Bearer token
        token = await _get_acled_token(c)
        if not token:
            return [], ACLED_READ
        
        # Build request parameters
        params = {
            "iso": country_code.upper()[:3],
            "event_date": f"{date_from}|{date_to}",
            "event_date_where": "BETWEEN",
            "limit": 500,
        }
        
        # Make API call with Bearer token
        headers = {"Authorization": f"Bearer {token}"}
        r = await c.get(ACLED_READ, params=params, headers=headers)
        r.raise_for_status()
        data = r.json()
        built = str(r.request.url).replace(token, "REDACTED")
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
