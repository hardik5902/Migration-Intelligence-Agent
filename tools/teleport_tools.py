"""
Urban quality-of-life scores.

NOTE: The Teleport public API (api.teleport.org) is no longer reachable — the
domain no longer resolves (DNS failure). This module now derives equivalent
quality-of-life category scores from World Bank indicators, which are more
authoritative and reliably available.

Category mapping (mirrors Teleport's category names for downstream compatibility):
  Healthcare     <- SH.XPD.CHEX.GD.ZS (health expenditure % GDP)
                    SH.MED.PHYS.ZS     (physicians per 1 000 people)
  Education      <- SE.XPD.TOTL.GD.ZS (education expenditure % GDP)
  Safety         <- PV.EST             (WGI political stability / no violence)
  Economy        <- NY.GDP.PCAP.CD     (GDP per capita USD)
  Cost of Living <- FP.CPI.TOTL.ZG    (CPI inflation — inverse proxy)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import httpx

from tools.worldbank_tools import get_indicator

# Teleport category name -> (WB indicator, score direction, scale_max)
# direction "higher_is_better" or "lower_is_better"
_CATEGORIES: list[tuple[str, str, str, float]] = [
    ("Healthcare",     "SH.XPD.CHEX.GD.ZS", "higher_is_better", 20.0),
    ("Healthcare",     "SH.MED.PHYS.ZS",     "higher_is_better",  8.0),
    ("Education",      "SE.XPD.TOTL.GD.ZS",  "higher_is_better", 15.0),
    ("Safety",         "PV.EST",              "higher_is_better",  2.5),   # WGI range ~-2.5 to +2.5
    ("Economy",        "NY.GDP.PCAP.CD",      "higher_is_better", 80000.0),
    ("Cost of Living", "FP.CPI.TOTL.ZG",      "lower_is_better",  30.0),  # low inflation = affordable
]


def _normalize(value: float, scale_max: float, higher_is_better: bool) -> float:
    """Map raw value to a 0-10 score."""
    ratio = max(0.0, min(value / scale_max, 1.0))
    score = ratio * 10.0
    if not higher_is_better:
        score = 10.0 - score
    return round(score, 2)


async def get_city_scores(
    city_or_country_hint: str,
    country_iso3: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """
    Return Teleport-compatible quality-of-life category scores for a country.

    Scores are derived from World Bank indicators instead of the defunct
    Teleport API. Each category score is on a 0-10 scale.
    """
    code = (country_iso3 or city_or_country_hint).strip().upper()
    # Use the most recent 5 years; take the latest non-null value per indicator.
    year_to = datetime.now(timezone.utc).year
    year_from = year_to - 5

    own = client is None
    c = client or httpx.AsyncClient(timeout=60.0)

    rows: list[dict[str, Any]] = []
    endpoint_url = f"https://api.worldbank.org/v2/country/{code}/indicator/{{indicator}}?format=json&mrv=5"

    try:
        # Gather all indicator fetches concurrently
        tasks = [
            get_indicator(code, ind, year_from, year_to, c)
            for _cat, ind, _dir, _scale in _CATEGORIES
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Accumulate one score-row per Teleport category (average if multiple indicators)
        category_values: dict[str, list[float]] = {}
        category_urls: dict[str, str] = {}

        for (cat, ind, direction, scale), result in zip(_CATEGORIES, results):
            if isinstance(result, Exception):
                print(f"[Teleport/WB] {ind} failed: {result}")
                continue
            wb_rows, url, err = result
            if err:
                print(f"[Teleport/WB] {ind} error: {err}")
            if not wb_rows:
                continue
            # Latest non-null value
            latest = max(wb_rows, key=lambda r: r.get("year", 0))
            raw = latest["value"]
            score = _normalize(raw, scale, direction == "higher_is_better")
            category_values.setdefault(cat, []).append(score)
            category_urls.setdefault(cat, url)

        fetched_at = datetime.now(timezone.utc).isoformat()
        for cat, scores in category_values.items():
            avg_score = round(sum(scores) / len(scores), 2)
            rows.append(
                {
                    "country": code,
                    "category": cat,
                    "score_out_of_10": avg_score,
                    "slug": code.lower(),
                    "data_source": "World Bank API (Teleport substitute)",
                    "source_api": "World Bank API",
                    "endpoint_url": category_urls.get(cat, endpoint_url),
                    "fetched_at": fetched_at,
                }
            )

        print(f"[Teleport/WB] Built {len(rows)} category scores for {code} via World Bank")

    except Exception as exc:
        print(f"[Teleport/WB] Unexpected error for {code}: {exc}")
        rows = []
    finally:
        if own:
            await c.aclose()

    used_url = rows[0]["endpoint_url"] if rows else endpoint_url
    return rows, used_url
