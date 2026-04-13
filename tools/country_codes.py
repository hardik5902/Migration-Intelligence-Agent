"""Map common country names to World Bank / UN style codes."""

from __future__ import annotations

import re
from difflib import get_close_matches

# Lowercase name -> ISO 3166-1 alpha-3 (World Bank uses alpha-3 in URLs)
_NAME_TO_ISO3: dict[str, str] = {
    "venezuela": "VEN",
    "syria": "SYR",
    "syrian arab republic": "SYR",
    "zimbabwe": "ZWE",
    "sudan": "SDN",
    "south sudan": "SSD",
    "afghanistan": "AFG",
    "ukraine": "UKR",
    "colombia": "COL",
    "turkey": "TUR",
    "türkiye": "TUR",
    "germany": "DEU",
    "france": "FRA",
    "india": "IND",
    "pakistan": "PAK",
    "bangladesh": "BGD",
    "united states": "USA",
    "usa": "USA",
    "united kingdom": "GBR",
    "uk": "GBR",
    "brazil": "BRA",
    "mexico": "MEX",
    "nigeria": "NGA",
    "ethiopia": "ETH",
    "somalia": "SOM",
    "yemen": "YEM",
    "iraq": "IRQ",
    "iran": "IRN",
    "lebanon": "LBN",
    "jordan": "JOR",
    "peru": "PER",
    "ecuador": "ECU",
    "chile": "CHL",
    "argentina": "ARG",
    "china": "CHN",
    "russia": "RUS",
    "myanmar": "MMR",
    "burma": "MMR",
    "haiti": "HTI",
    "cuba": "CUB",
    "new zealand": "NZL",
    "finland": "FIN",
    "norway": "NOR",
    "sweden": "SWE",
    "spain": "ESP",
    "italy": "ITA",
    "canada": "CAN",
    "australia": "AUS",
    "japan": "JPN",
    "south korea": "KOR",
    "korea": "KOR",
    "antarctica": "ATA",
}


def normalize_name(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def country_name_to_iso3(name: str) -> str | None:
    if not name:
        return None
    n = normalize_name(name)
    if n in _NAME_TO_ISO3:
        return _NAME_TO_ISO3[n]
    matches = get_close_matches(n, list(_NAME_TO_ISO3.keys()), n=1, cutoff=0.75)
    if matches:
        return _NAME_TO_ISO3[matches[0]]
    if len(name.strip()) == 3 and name.isalpha():
        return name.strip().upper()
    return None


def iso3_to_iso2(iso3: str) -> str | None:
    m = {
        "USA": "US",
        "GBR": "GB",
        "VEN": "VE",
        "SYR": "SY",
        "ZWE": "ZW",
        "SDN": "SD",
        "IND": "IN",
        "COL": "CO",
        "DEU": "DE",
        "FRA": "FR",
        "UKR": "UA",
        "AFG": "AF",
        "ESP": "ES",
        "ITA": "IT",
        "CAN": "CA",
        "AUS": "AU",
        "JPN": "JP",
        "KOR": "KR",
        "NZL": "NZ",
        "FIN": "FI",
        "NOR": "NO",
        "SWE": "SE",
        "CHE": "CH",
        "NLD": "NL",
        "DNK": "DK",
        "IRL": "IE",
        "SGP": "SG",
        "ATA": "AQ",
    }
    return m.get(iso3.upper())


def iso3_to_name(iso3: str) -> str | None:
    code = iso3.upper()
    for name, value in _NAME_TO_ISO3.items():
        if value == code and len(name) > 3:
            return " ".join(part.capitalize() for part in name.split())
    return None
