"""Teleport public API for urban quality-of-life scores."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

TELEPORT = "https://api.teleport.org/api"


async def _urban_slug_for_country(capital_query: str, client: httpx.AsyncClient) -> str | None:
    r = await client.get(f"{TELEPORT}/cities/", params={"search": capital_query})
    r.raise_for_status()
    data = r.json()
    href = None
    if isinstance(data, dict) and data.get("_embedded", {}).get("city:search-results"):
        first = data["_embedded"]["city:search-results"][0]
        href = first.get("_links", {}).get("city:item", {}).get("href")
    if not href:
        return None
    r2 = await client.get(href)
    r2.raise_for_status()
    city = r2.json()
    ua = city.get("_links", {}).get("city:urban_area", {}).get("href")
    if not ua:
        return None
    # href ends with geoname_id — fetch urban area for slug
    r3 = await client.get(ua)
    r3.raise_for_status()
    ua_data = r3.json()
    slug = ua_data.get("slug")
    return slug


async def get_city_scores(
    city_or_country_hint: str,
    client: httpx.AsyncClient | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Fetch Teleport scores for the best-matching urban area."""
    own = client is None
    c = client or httpx.AsyncClient(timeout=45.0)
    rows: list[dict[str, Any]] = []
    url = f"{TELEPORT}/urban_areas/slug:unknown/scores/"
    try:
        slug = await _urban_slug_for_country(city_or_country_hint, c)
        if not slug:
            return [], url
        url = f"{TELEPORT}/urban_areas/slug:{slug}/scores/"
        r = await c.get(url)
        r.raise_for_status()
        data = r.json()
        cats = data.get("categories", [])
        for cat in cats:
            if not isinstance(cat, dict):
                continue
            rows.append(
                {
                    "category": cat.get("name", ""),
                    "score_out_of_10": float(cat.get("score_out_of_10") or 0),
                    "slug": slug,
                    "source_api": "Teleport API",
                    "endpoint_url": url,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                }
            )
    except Exception:
        rows = []
    finally:
        if own:
            await c.aclose()
    return rows, url
