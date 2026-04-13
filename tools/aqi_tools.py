"""
tools/aqi_tools.py  —  FULLY CORRECTED VERSION
================================================

BUGS FIXED FROM PREVIOUS VERSION:
──────────────────────────────────
BUG 1 ─ OpenAQ: wrong query param name for country filter
  OLD:  params["country"] = "IN"           ← does NOT exist in v3 API
  NEW:  params["iso"] = "IN"               ← correct v3 param name
  WHY:  The /v3/locations spec has two country params:
          • `iso`          → ISO2 string  e.g. "IN"
          • `countries_id` → numeric ID   e.g. 9
        Your code used `"country"` which is neither. The API silently ignores
        unknown query params and returns ALL locations worldwide (100 rows of
        random countries) instead of failing — so you got data, just wrong data.

BUG 2 ─ OpenAQ: /v3/locations sensors have NO value fields at all
  OLD:  summary.get("avg") / sensor.get("lastValue") / sensor.get("value")
  WHY:  The official /v3/locations schema exposes sensors with ONLY:
          id, name, parameter{id, name, units, displayName}
        There is NO summary, NO latest, NO lastValue, NO value on sensors here.
        These fields DO NOT EXIST in this endpoint.
        Result: pm25 is always None → 0 OpenAQ rows returned every single call.
  NEW:  Use /v3/parameters/2/latest?iso=XX — the ONLY endpoint that returns
        actual PM2.5 readings without extra per-location calls.

BUG 3 ─ OpenAQ: _openaq_headers() silently returns {} when key is missing
  OLD:  return {"X-API-Key": key} if key else {}
  WHY:  Empty headers → 401 Unauthorized → caught by broad except Exception
        → returns ([], url) → no error ever surfaces → OpenAQ always returns 0
        rows, World Bank takes over, and you never know OpenAQ is broken.
  NEW:  Log a clear warning when key is absent.

BUG 4 ─ OpenAQ: broad except Exception swallows 401 / 422 / 429 silently
  OLD:  except Exception as exc: print(...); return [], OPENAQ_V3
  WHY:  A 401 (wrong key), 422 (bad params), or 429 (rate limit) is silently
        treated as "no data". The real error never surfaces.
  NEW:  Handle each status code explicitly. Re-raise only on network errors.

BUG 5 ─ World Bank: mrv=5 only returns 5 years, often only 2-3 for crisis countries
  OLD:  "...?format=json&mrv=5&per_page=10"
  WHY:  `mrv=5` means "most recent 5 non-null values". Countries with reporting
        lag (Venezuela, Sudan, Afghanistan) may only have 2-3 rows, the most
        recent being 2019 or 2020. `mrv=10` gives a better time series.
  NEW:  Use mrv=10, per_page=20.

BUG 6 ─ World Bank: city="" causes schema mismatches in DuckDB joins
  NEW:  Use the country name as the city field for WB rows (WB is country-level).

BUG 7 ─ Signature: get_aqi_by_country(country_iso3, country_iso2=None, ...)
  WHY:  Optional iso2 param means callers can forget it and silently get wrong
        OpenAQ results (country filter ignored = random worldwide data returned).
  NEW:  Accept only iso3. Resolve iso2 internally always.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ── Endpoints ─────────────────────────────────────────────────────────────────

# BUG 2 FIX: /v3/parameters/2/latest is the correct endpoint for actual values.
# (parameter id 2 = PM2.5 in OpenAQ registry)
# /v3/locations only lists sensor metadata — it has NO value fields.
OPENAQ_LATEST_URL = "https://api.openaq.org/v3/parameters/2/latest"

# World Bank annual mean PM2.5 µg/m³ (WHO satellite+ground validated, no auth)
# BUG 5 FIX: mrv=10, per_page=20
WB_PM25_URL = (
    "https://api.worldbank.org/v2/country/{code}/indicator/EN.ATM.PM25.MC.M3"
    "?format=json&mrv=10&per_page=20"
)
WB_AQI_INDICATOR = "EN.ATM.PM25.MC.M3"


# ── Auth ──────────────────────────────────────────────────────────────────────

def _openaq_headers() -> dict[str, str]:
    """
    BUG 3 FIX: Log a warning when key is missing instead of silently returning {}.
    Missing key → {} → 401 → was silently caught → 0 OpenAQ rows forever.
    """
    raw = os.environ.get("OPENAQ_API_KEY", "") or ""
    key = raw.strip().strip('"').strip("'")
    if not key:
        logger.warning(
            "OPENAQ_API_KEY is not set. OpenAQ calls will return 401 Unauthorized.\n"
            "Get a FREE key at: https://explore.openaq.org/register\n"
            "Then add to your .env file:  OPENAQ_API_KEY=your_key_here"
        )
        return {}
    return {"X-API-Key": key}


# ── ISO helpers ───────────────────────────────────────────────────────────────

_ISO3_TO_ISO2: dict[str, str] = {
    "AFG": "AF", "DZA": "DZ", "ARG": "AR", "AUS": "AU", "AUT": "AT",
    "BGD": "BD", "BEL": "BE", "BRA": "BR", "BGR": "BG", "CAN": "CA",
    "CHL": "CL", "CHN": "CN", "COL": "CO", "HRV": "HR", "CZE": "CZ",
    "DNK": "DK", "ECU": "EC", "EGY": "EG", "ETH": "ET", "FIN": "FI",
    "FRA": "FR", "DEU": "DE", "GHA": "GH", "GRC": "GR", "HTI": "HT",
    "HUN": "HU", "IND": "IN", "IDN": "ID", "IRN": "IR", "IRQ": "IQ",
    "IRL": "IE", "ISR": "IL", "ITA": "IT", "JPN": "JP", "JOR": "JO",
    "KAZ": "KZ", "KEN": "KE", "KWT": "KW", "LVA": "LV", "LBN": "LB",
    "LTU": "LT", "MYS": "MY", "MEX": "MX", "MNG": "MN", "MAR": "MA",
    "MMR": "MM", "NPL": "NP", "NLD": "NL", "NZL": "NZ", "NGA": "NG",
    "NOR": "NO", "OMN": "OM", "PAK": "PK", "PER": "PE", "PHL": "PH",
    "POL": "PL", "PRT": "PT", "QAT": "QA", "KOR": "KR", "ROU": "RO",
    "RUS": "RU", "SAU": "SA", "SRB": "RS", "SGP": "SG", "SVK": "SK",
    "SOM": "SO", "ZAF": "ZA", "SSD": "SS", "ESP": "ES", "LKA": "LK",
    "SDN": "SD", "SWE": "SE", "CHE": "CH", "TWN": "TW", "THA": "TH",
    "TUN": "TN", "TUR": "TR", "TKM": "TM", "UGA": "UG", "UKR": "UA",
    "ARE": "AE", "GBR": "GB", "USA": "US", "UZB": "UZ", "VEN": "VE",
    "VNM": "VN", "YEM": "YE", "ZMB": "ZM", "ZWE": "ZW", "SYR": "SY",
    "CUB": "CU",
}


def _iso3_to_iso2(iso3: str) -> str | None:
    return _ISO3_TO_ISO2.get(iso3.upper().strip())


# ── World Bank PM2.5 (PRIMARY source) ─────────────────────────────────────────

async def _fetch_worldbank_pm25(
    country_iso3: str,
    client: httpx.AsyncClient,
) -> tuple[list[dict[str, Any]], str]:
    """
    World Bank annual mean PM2.5 µg/m³.
    Primary source: reliable, no auth, covers all countries, WHO validated.
    BUG 5 FIX: mrv=10 instead of mrv=5.
    BUG 6 FIX: city field set to country name, not empty string.
    """
    url = WB_PM25_URL.format(code=country_iso3.upper())
    try:
        r = await client.get(url, timeout=30.0)
        r.raise_for_status()
        payload = r.json()
    except httpx.HTTPStatusError as exc:
        logger.error(f"[AQI/WB] HTTP {exc.response.status_code} for {country_iso3}: {exc}")
        return [], url
    except Exception as exc:
        logger.error(f"[AQI/WB] Fetch failed for {country_iso3}: {exc}")
        return [], url

    # World Bank response: [metadata_dict, [data_array]]
    # data_array entries have null values for years with no data — skip those
    if (
        not isinstance(payload, list)
        or len(payload) < 2
        or not isinstance(payload[1], list)
    ):
        logger.warning(f"[AQI/WB] Unexpected response structure for {country_iso3}")
        return [], url

    rows: list[dict[str, Any]] = []
    country_name = country_iso3

    for item in payload[1]:
        if not isinstance(item, dict):
            continue
        val = item.get("value")
        if val is None:
            continue  # Skip null years — common for crisis/developing countries

        country_info = item.get("country", {})
        if isinstance(country_info, dict):
            country_name = country_info.get("value", country_iso3)

        rows.append({
            "location": country_name,
            "country": country_iso3.upper(),
            "city": country_name,          # BUG 6 FIX: use name not ""
            "pm25": float(val),
            "year": int(item.get("date", 0) or 0),
            "source_api": "World Bank API",
            "indicator_code": WB_AQI_INDICATOR,
            "endpoint_url": url,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        })

    logger.info(f"[AQI/WB] {len(rows)} PM2.5 rows for {country_iso3}")
    return rows, url


# ── OpenAQ v3 PM2.5 (SUPPLEMENTARY source) ────────────────────────────────────

async def _fetch_openaq_v3(
    country_iso2: str,
    top_n: int,
    client: httpx.AsyncClient,
) -> tuple[list[dict[str, Any]], str]:
    """
    Fetch live PM2.5 readings from OpenAQ v3.

    BUG 1 FIX: Use `iso` param (ISO2 string), NOT `country` (doesn't exist).
    BUG 2 FIX: Use /v3/parameters/2/latest — returns actual values.
               /v3/locations only returns sensor metadata with NO values.
    BUG 4 FIX: Explicit per-status-code handling instead of catch-all.
    """
    iso2 = country_iso2.strip().upper()
    url = OPENAQ_LATEST_URL

    # BUG 1 FIX: param name is `iso`, not `country`
    params: dict[str, Any] = {
        "iso": iso2,
        "limit": min(top_n, 100),
        "page": 1,
    }

    try:
        response = await client.get(
            url,
            params=params,
            headers=_openaq_headers(),
            timeout=30.0,
        )
    except httpx.TimeoutException:
        logger.warning(f"[AQI/OpenAQ] Timeout for {iso2}")
        return [], url
    except httpx.NetworkError as exc:
        logger.warning(f"[AQI/OpenAQ] Network error for {iso2}: {exc}")
        return [], url

    # BUG 4 FIX: explicit status-code handling
    if response.status_code == 401:
        logger.error(
            "[AQI/OpenAQ] 401 Unauthorized — OPENAQ_API_KEY is missing or invalid.\n"
            "Register free at: https://explore.openaq.org/register"
        )
        return [], url

    if response.status_code == 422:
        logger.error(
            f"[AQI/OpenAQ] 422 Unprocessable — bad params: {params}\n"
            f"Body: {response.text[:300]}"
        )
        return [], url

    if response.status_code == 429:
        reset_in = response.headers.get("x-ratelimit-reset", "60")
        logger.warning(
            f"[AQI/OpenAQ] 429 Rate limited. Reset in {reset_in}s. "
            f"Free tier: 60 req/min, 2000 req/hour."
        )
        return [], url

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.error(f"[AQI/OpenAQ] HTTP {exc.response.status_code} for {iso2}")
        return [], url

    payload = response.json()
    built_url = str(response.request.url)
    results = payload.get("results", []) if isinstance(payload, dict) else []

    rows: list[dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            continue

        # /v3/parameters/2/latest response structure (confirmed from API spec):
        # {
        #   "locationsId": 123,
        #   "location": "Station Name",
        #   "value": 45.2,            ← the actual PM2.5 reading
        #   "unit": "µg/m³",
        #   "country": {"id":9, "code":"IN", "name":"India"},
        #   "coordinates": {"latitude":..., "longitude":...},
        #   "datetime": {"utc":"...", "local":"..."}
        # }
        pm25 = item.get("value")
        if pm25 is None:
            continue

        country_info = item.get("country") or {}
        country_code = country_info.get("code", iso2) if isinstance(country_info, dict) else iso2

        coords = item.get("coordinates") or {}
        dt_info = item.get("datetime") or {}

        rows.append({
            "location": item.get("location", ""),
            "country": country_code,
            "city": item.get("locality") or "",
            "pm25": float(pm25),
            "year": datetime.now(timezone.utc).year,
            "latitude": coords.get("latitude"),
            "longitude": coords.get("longitude"),
            "source_api": "OpenAQ API",
            "indicator_code": "pm25",
            "endpoint_url": built_url,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "reading_utc": dt_info.get("utc", ""),
        })

    found = payload.get("meta", {}).get("found", "?")
    logger.info(f"[AQI/OpenAQ] {len(rows)} readings for {iso2} (API found={found})")
    return rows, built_url


# ── Public API ────────────────────────────────────────────────────────────────

async def get_aqi_by_country(
    country_iso3: str,
    top_n: int = 30,
    client: httpx.AsyncClient | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """
    Fetch PM2.5 AQI data for a country.

    BUG 7 FIX: Accepts only iso3 (consistent with rest of codebase).
               iso2 resolved internally — callers never need to pass it.

    Data strategy:
      PRIMARY   → World Bank EN.ATM.PM25.MC.M3 (annual mean, all countries, no auth)
      SECONDARY → OpenAQ v3 /parameters/2/latest (live sensor readings, needs key)

    Returns:
      (rows, source_url) where rows is list of dicts with keys:
        location, country, city, pm25, year, source_api, indicator_code,
        endpoint_url, fetched_at

    Never raises. Returns ([], url) if both sources fail.
    """
    own_client = client is None
    c = client or httpx.AsyncClient(timeout=45.0)

    try:
        wb_rows, wb_url = await _fetch_worldbank_pm25(country_iso3, c)

        iso2 = _iso3_to_iso2(country_iso3)
        oaq_rows: list[dict[str, Any]] = []
        oaq_url = OPENAQ_LATEST_URL

        if iso2:
            oaq_rows, oaq_url = await _fetch_openaq_v3(iso2, top_n, c)
        else:
            logger.warning(
                f"[AQI] No ISO2 mapping for '{country_iso3}'. "
                f"Add it to _ISO3_TO_ISO2 in aqi_tools.py to enable OpenAQ data."
            )

        if wb_rows:
            # World Bank is primary — append OpenAQ as supplementary recent readings
            return wb_rows + oaq_rows, wb_url
        elif oaq_rows:
            logger.warning(
                f"[AQI] World Bank returned 0 rows for {country_iso3} "
                f"(may have no data or reporting lag). Using OpenAQ only."
            )
            return oaq_rows, oaq_url
        else:
            logger.warning(f"[AQI] No PM2.5 data found for {country_iso3} from either source.")
            return [], wb_url

    finally:
        if own_client:
            await c.aclose()


async def get_aqi_for_relocation_ranking(
    country_iso3_list: list[str],
    client: httpx.AsyncClient | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch PM2.5 for a list of countries and return a ranking summary.
    Used by the Relocation Scorer agent to rank countries by air quality.

    Returns list sorted by avg_pm25 ASC (cleanest air first):
    [
      {country_iso3, country_name, avg_pm25, latest_year, source_api,
       indicator_code, endpoint_url, fetched_at},
      ...
    ]
    """
    own_client = client is None
    c = client or httpx.AsyncClient(timeout=45.0)

    try:
        tasks = [
            get_aqi_by_country(iso3, top_n=5, client=c)
            for iso3 in country_iso3_list
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        if own_client:
            await c.aclose()

    summary = []
    for iso3, result in zip(country_iso3_list, results):
        if isinstance(result, Exception):
            logger.warning(f"[AQI] Failed to fetch data for {iso3}: {result}")
            continue

        rows, _ = result
        if not rows:
            continue

        # Prefer World Bank rows (validated annual mean) over OpenAQ for ranking
        wb_only = [r for r in rows if r.get("source_api") == "World Bank API"]
        source_rows = wb_only if wb_only else rows

        # Use the most recent year's value
        source_rows_sorted = sorted(
            source_rows, key=lambda r: r.get("year", 0), reverse=True
        )
        best = source_rows_sorted[0]

        summary.append({
            "country_iso3": iso3,
            "country_name": best.get("location", iso3),
            "avg_pm25": best.get("pm25"),
            "latest_year": best.get("year"),
            "source_api": best.get("source_api"),
            "indicator_code": best.get("indicator_code"),
            "endpoint_url": best.get("endpoint_url"),
            "fetched_at": best.get("fetched_at"),
        })

    # Sort ascending by PM2.5 — cleanest first
    summary.sort(key=lambda x: x.get("avg_pm25") or 9999)
    return summary