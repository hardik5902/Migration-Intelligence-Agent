"""NewsAPI + GDELT helpers."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

NEWS_EVERYTHING = "https://newsapi.org/v2/everything"
GDELT_DOC = "https://api.gdeltproject.org/api/v2/doc/doc"


async def get_country_news(
    country: str,
    from_date: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Headlines from NewsAPI (requires NEWS_API_KEY)."""
    key = os.environ.get("NEWS_API_KEY", "")
    if not key:
        return [], NEWS_EVERYTHING
    if not from_date:
        from_date = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
    params = {
        "q": country,
        "from": from_date,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 20,
        "apiKey": key,
    }
    own = client is None
    c = client or httpx.AsyncClient(timeout=45.0)
    rows: list[dict[str, Any]] = []
    built = NEWS_EVERYTHING
    try:
        r = await c.get(NEWS_EVERYTHING, params=params)
        r.raise_for_status()
        data = r.json()
        built = str(r.request.url).replace(key, "REDACTED")
        for art in data.get("articles", []) or []:
            if not isinstance(art, dict):
                continue
            rows.append(
                {
                    "country": country,
                    "title": art.get("title") or "",
                    "source": (art.get("source") or {}).get("name", "")
                    if isinstance(art.get("source"), dict)
                    else "",
                    "published_at": art.get("publishedAt", ""),
                    "description": art.get("description") or "",
                    "url": art.get("url") or "",
                    "source_api": "NewsAPI",
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


async def get_gdelt_sentiment(
    country_query: str,
    client: httpx.AsyncClient | None = None,
) -> tuple[dict[str, Any], str]:
    """Aggregate tone and timeline from GDELT doc API (no key)."""
    params = {
        "query": country_query,
        "mode": "artlist",
        "maxrecords": 50,
        "format": "json",
    }
    own = client is None
    c = client or httpx.AsyncClient(timeout=45.0)
    out: dict[str, Any] = {
        "avg_tone": 0.0,
        "article_count": 0,
        "timeline": [],
    }
    built = GDELT_DOC
    try:
        r = await c.get(GDELT_DOC, params=params)
        r.raise_for_status()
        data = r.json()
        built = str(r.request.url)
        tones: list[float] = []
        timeline_rows: list[dict[str, Any]] = []
        articles = data.get("articles", []) if isinstance(data, dict) else []
        for article in articles:
            if not isinstance(article, dict):
                continue
            tone_raw = article.get("tone")
            try:
                tone = float(tone_raw)
            except Exception:
                tone = None
            if tone is not None:
                tones.append(tone)
            date_key = str(article.get("seendate") or article.get("date") or "")[:8]
            if date_key:
                timeline_rows.append(
                    {
                        "date": date_key,
                        "tone": tone,
                        "source": article.get("domain", ""),
                        "title": article.get("title", ""),
                    }
                )
        if tones:
            out["avg_tone"] = sum(tones) / len(tones)
            out["article_count"] = len(tones)
        out["timeline"] = timeline_rows
        out["source_api"] = "GDELT API"
        out["endpoint_url"] = built
        out["fetched_at"] = datetime.now(timezone.utc).isoformat()
    except Exception:
        out["endpoint_url"] = built
    finally:
        if own:
            await c.aclose()
    return out, built


async def gdelt_sentiment_score(
    country_query: str,
    client: httpx.AsyncClient | None = None,
) -> float:
    """Deterministic scalar tone for tool callers."""
    d, _ = await get_gdelt_sentiment(country_query, client)
    return float(d.get("avg_tone") or 0.0)
