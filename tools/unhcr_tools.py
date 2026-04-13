"""UNHCR population API (async httpx)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
from tools.country_codes import iso3_to_iso2, iso3_to_name

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
        "coa_all": "true",
        "cf_type": "ISO",
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
    if not items:
        try:
            # helpful for debugging: print truncated payload when no items found
            txt = str(payload)
            print(f"[UNHCR] payload (truncated): {txt[:1000]}")
        except Exception:
            pass
    
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
    
    # If no rows were found, attempt a ISO2 fallback (some UNHCR endpoints
    # may accept alpha-2 codes or country name variants). Try once more
    # with iso2 and then with a friendly name.
    if not rows:
        iso2 = iso3_to_iso2(code) or ""
        tried = []
        if iso2:
            try:
                params2 = {**params, "coo": iso2}
                r2 = await c.get(url, params=params2, follow_redirects=True)
                r2.raise_for_status()
                payload2 = r2.json()
                built2 = str(r2.request.url)
                items2 = payload2.get("items") or payload2.get("data") or []
                print(f"[UNHCR] Fallback ISO2 fetch returned {len(items2)} items for {iso2}")
                tried.append(built2)
                for it in items2:
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
                            "endpoint_url": built2,
                            "fetched_at": datetime.now(timezone.utc).isoformat(),
                        }
                    )
            except Exception as exc3:
                print(f"[UNHCR] ISO2 fallback failed for {iso2}: {exc3}")
        # Try friendly name
        if not rows:
            friendly = iso3_to_name(code) or ""
            if friendly:
                try:
                    params3 = {**params, "coo": friendly}
                    r3 = await c.get(url, params=params3, follow_redirects=True)
                    r3.raise_for_status()
                    payload3 = r3.json()
                    built3 = str(r3.request.url)
                    items3 = payload3.get("items") or payload3.get("data") or []
                    print(f"[UNHCR] Fallback name fetch returned {len(items3)} items for {friendly}")
                    tried.append(built3)
                    for it in items3:
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
                                "endpoint_url": built3,
                                "fetched_at": datetime.now(timezone.utc).isoformat(),
                            }
                        )
                except Exception as exc4:
                    print(f"[UNHCR] Friendly-name fallback failed for {friendly}: {exc4}")
        if tried:
            built_url = tried[-1]

    return rows, built_url
