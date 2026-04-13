
"""AQI / PM2.5 data.

Primary source  : World Bank indicator EN.ATM.PM25.MC.M3
                  (WHO-validated annual mean PM2.5 µg/m³, all countries, no auth)
Secondary source: OpenAQ V3 /v3/locations with corrected parameter-dict parsing
                  (live sensor readings, used to supplement / validate WB data)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import httpx

OPENAQ_V3 = "https://api.openaq.org/v3/locations"
WB_PM25_URL = "https://api.worldbank.org/v2/country/{code}/indicator/EN.ATM.PM25.MC.M3?format=json&mrv=5&per_page=10"
WB_AQI_INDICATOR = "EN.ATM.PM25.MC.M3"


def _openaq_headers() -> dict[str, str]:
    key = os.environ.get("OPENAQ_API_KEY", "").strip()
    return {"X-API-Key": key} if key else {}


async def _fetch_worldbank_pm25(
    country_iso3: str,
    client: httpx.AsyncClient,
) -> tuple[list[dict[str, Any]], str]:
    """World Bank annual mean PM2.5 µg/m³ — reliable, all countries, no auth."""
    url = WB_PM25_URL.format(code=country_iso3.upper())
    try:
        r = await client.get(url)
        r.raise_for_status()
        payload = r.json()
    except Exception as exc:
        print(f"[AQI/WB] PM2.5 fetch failed for {country_iso3}: {exc}")
        return [], url

    rows: list[dict[str, Any]] = []
    if isinstance(payload, list) and len(payload) > 1 and isinstance(payload[1], list):
        for item in payload[1]:
            if not isinstance(item, dict):
                continue
            val = item.get("value")
            if val is None:
                continue
            country_info = item.get("country", {})
            rows.append(
                {
                    "location": country_info.get("value", country_iso3) if isinstance(country_info, dict) else country_iso3,
                    "country": country_iso3.upper(),
                    "city": "",
                    "pm25": float(val),
                    "year": int(item.get("date", 0) or 0),
                    "source_api": "World Bank API (EN.ATM.PM25.MC.M3)",
                    "indicator_code": WB_AQI_INDICATOR,
                    "endpoint_url": url,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                }
            )
    print(f"[AQI/WB] Got {len(rows)} PM2.5 rows for {country_iso3}")
    return rows, url


async def _fetch_openaq_v3(
    country_iso2: str,
    top_n: int,
    client: httpx.AsyncClient,
) -> tuple[list[dict[str, Any]], str]:
    """
    OpenAQ V3 /v3/locations with corrected parameter-dict parsing.
    In V3 the 'parameter' field inside each sensor is a DICT, not a string.
    The summary.avg from /v3/locations is often null — we skip nulls here
    and let World Bank data be the primary source.
    """
    params: dict[str, Any] = {
        "limit": min(top_n, 100),
        "parameters_id": 2,  # 2 = PM2.5 in OpenAQ parameter registry
    }
    q = country_iso2.strip().upper()
    if len(q) == 2:
        params["country"] = q
    else:
        params["query"] = country_iso2

    try:
        response = await client.get(OPENAQ_V3, params=params, headers=_openaq_headers())
        response.raise_for_status()
    except Exception as exc:
        print(f"[AQI/OpenAQ] V3 fetch failed for {country_iso2}: {exc}")
        return [], OPENAQ_V3

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

        pm25 = None
        for sensor in item.get("sensors", []):
            if not isinstance(sensor, dict):
                continue

            # V3: parameter is a DICT {"id": 2, "name": "pm25", "units": "µg/m³", ...}
            param_info = sensor.get("parameter", {})
            if isinstance(param_info, dict):
                param_name = param_info.get("name", "")
            else:
                param_name = str(param_info or "")

            if param_name.lower() != "pm25":
                continue

            # summary.avg is often null in /v3/locations — try all value fields
            summary = sensor.get("summary") or {}
            pm25 = summary.get("avg") if isinstance(summary, dict) else None
            if pm25 is None:
                # try latest measurement fields
                latest = sensor.get("latest") or {}
                pm25 = (
                    latest.get("value")
                    or sensor.get("lastValue")
                    or sensor.get("value")
                )
            break

        if pm25 is not None:
            rows.append(
                {
                    "location": item.get("name", ""),
                    "country": country or q,
                    "city": item.get("city", "") if isinstance(item.get("city"), str) else "",
                    "pm25": float(pm25),
                    "year": datetime.now(timezone.utc).year,
                    "source_api": "OpenAQ API",
                    "indicator_code": "pm25",
                    "endpoint_url": built,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                }
            )

    print(f"[AQI/OpenAQ] V3 returned {len(rows)} rows with valid PM2.5 for {country_iso2}")
    return rows, built


async def get_aqi_by_country(
    country_iso3: str,
    country_iso2: str | None = None,
    top_n: int = 30,
    client: httpx.AsyncClient | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """
    Fetch PM2.5 AQI data for a country.

    Strategy:
      1. World Bank EN.ATM.PM25.MC.M3 — reliable annual mean for all countries (primary)
      2. OpenAQ V3 live sensor readings — supplementary recent data (may return 0 rows if
         summary is null, which is normal for the /v3/locations endpoint)

    Returns the World Bank rows if they exist, supplemented by OpenAQ if available.
    Falls back to OpenAQ-only if World Bank returns nothing.
    """
    own = client is None
    c = client or httpx.AsyncClient(timeout=45.0)
    try:
        wb_rows, wb_url = await _fetch_worldbank_pm25(country_iso3, c)

        iso2 = country_iso2 or _iso3_to_iso2(country_iso3)
        oaq_rows: list[dict[str, Any]] = []
        oaq_url = OPENAQ_V3
        if iso2:
            oaq_rows, oaq_url = await _fetch_openaq_v3(iso2, top_n, c)

        # Primary: World Bank. Supplement with OpenAQ if WB returned nothing.
        if wb_rows:
            primary_url = wb_url
            # Attach any live OpenAQ readings as extra rows labelled distinctly
            return wb_rows + oaq_rows, primary_url
        elif oaq_rows:
            print(f"[AQI] World Bank returned 0 rows, using OpenAQ only for {country_iso3}")
            return oaq_rows, oaq_url
        else:
            print(f"[AQI] No PM2.5 data found for {country_iso3} from any source")
            return [], wb_url
    finally:
        if own:
            await c.aclose()


def _iso3_to_iso2(iso3: str) -> str | None:
    """Minimal ISO3→ISO2 mapping for OpenAQ country filter."""
    mapping = {
        "NZL": "NZ", "FIN": "FI", "NOR": "NO", "SWE": "SE", "CHE": "CH",
        "NLD": "NL", "DNK": "DK", "IRL": "IE", "SGP": "SG", "AUS": "AU",
        "CAN": "CA", "DEU": "DE", "FRA": "FR", "GBR": "GB", "JPN": "JP",
        "KOR": "KR", "USA": "US", "IND": "IN", "VEN": "VE", "SYR": "SY",
        "ZWE": "ZW", "SDN": "SD", "COL": "CO", "UKR": "UA", "AFG": "AF",
        "ESP": "ES", "ITA": "IT", "BRA": "BR", "MEX": "MX", "NGA": "NG",
        "ETH": "ET", "SOM": "SO", "YEM": "YE", "IRQ": "IQ", "IRN": "IR",
        "LBN": "LB", "JOR": "JO", "PER": "PE", "ECU": "EC", "CHL": "CL",
        "ARG": "AR", "CHN": "CN", "RUS": "RU", "MMR": "MM", "HTI": "HT",
        "CUB": "CU", "PAK": "PK", "BGD": "BD", "TUR": "TR",
    }
    return mapping.get(iso3.upper())
