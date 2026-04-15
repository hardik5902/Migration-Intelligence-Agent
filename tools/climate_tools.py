"""Open-Meteo historical climate (archive API)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

# Approximate capital coordinates for country-level climate pulls
_CAPITAL_LATLON: dict[str, tuple[float, float]] = {
    "VEN": (10.48, -66.90),
    "SYR": (33.51, 36.29),
    "ZWE": (-17.83, 31.05),
    "SDN": (15.50, 32.56),
    "UKR": (50.45, 30.52),
    "COL": (4.71, -74.07),
    "IND": (28.61, 77.21),
    "USA": (38.90, -77.04),
    "GBR": (51.51, -0.13),
    "DEU": (52.52, 13.41),
    "FRA": (48.86, 2.35),
    "AFG": (34.56, 69.20),
    "PAK": (33.69, 73.05),
    "BGD": (23.81, 90.41),
    "NGA": (9.08, 7.53),
    "ETH": (9.03, 38.75),
    "SOM": (2.05, 45.32),
    "YEM": (15.37, 44.19),
    "IRQ": (33.31, 44.37),
    "IRN": (35.69, 51.39),
    "LBN": (33.89, 35.50),
    "JOR": (31.95, 35.93),
    "PER": (-12.05, -77.04),
    "ECU": (-0.18, -78.47),
    "CHL": (-33.45, -70.67),
    "ARG": (-34.60, -58.38),
    "CHN": (39.90, 116.41),
    "RUS": (55.76, 37.62),
    "MMR": (16.87, 96.20),
    "HTI": (18.59, -72.31),
    "CUB": (23.13, -82.37),
    "NZL": (-41.29, 174.78),
    "FIN": (60.17, 24.94),
    "NOR": (59.91, 10.75),
    "SWE": (59.33, 18.07),
    "ESP": (40.42, -3.70),
    "ITA": (41.90, 12.48),
    "CAN": (45.42, -75.70),
    "AUS": (-35.28, 149.13),
    "JPN": (35.68, 139.76),
    "KOR": (37.57, 126.98),
    "BRA": (-15.80, -47.88),
    "MEX": (19.43, -99.13),
}


async def get_climate_data(
    country_code: str,
    year_from: int,
    year_to: int,
    client: httpx.AsyncClient | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Daily aggregates rolled to annual rows in returned list."""
    code = country_code.upper()[:3]
    latlon = _CAPITAL_LATLON.get(code, (20.0, 0.0))
    lat, lon = latlon
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{year_from}-01-01",
        "end_date": f"{year_to}-12-31",
        "daily": "temperature_2m_max,precipitation_sum",
    }
    own = client is None
    c = client or httpx.AsyncClient(timeout=12.0)
    rows: list[dict[str, Any]] = []
    built = ARCHIVE
    try:
        r = await c.get(ARCHIVE, params=params)
        r.raise_for_status()
        data = r.json()
        built = str(r.request.url)
        daily = data.get("daily", {})
        temps = daily.get("temperature_2m_max", []) or []
        prec = daily.get("precipitation_sum", []) or []
        dates = daily.get("time", []) or []
        by_year: dict[int, list[float]] = {}
        by_year_p: dict[int, list[float]] = {}
        for i, d in enumerate(dates):
            y = int(str(d)[:4])
            by_year.setdefault(y, []).append(float(temps[i]) if i < len(temps) else 0.0)
            by_year_p.setdefault(y, []).append(float(prec[i]) if i < len(prec) else 0.0)
        for y in sorted(by_year):
            tavg = sum(by_year[y]) / max(len(by_year[y]), 1)
            psum = sum(by_year_p.get(y, [0.0]))
            rows.append(
                {
                    "country": code,
                    "year": y,
                    "avg_daily_max_temp_c": tavg,
                    "annual_precipitation_mm": psum,
                    "source_api": "Open-Meteo Archive",
                    "endpoint_url": built,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        if rows:
            baseline = sum(row["avg_daily_max_temp_c"] for row in rows) / len(rows)
            for row in rows:
                row["avg_temp_anomaly_c"] = row["avg_daily_max_temp_c"] - baseline
                row["extreme_heat_days"] = 0
    except Exception:
        rows = []
    finally:
        if own:
            await c.aclose()
    return rows, built
