"""Shared constants for the country comparison pipeline."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Tool registry — every tool that can be selected, shown in the UI and
# fed to the LLM tool selector.
# ---------------------------------------------------------------------------

AVAILABLE_TOOLS: dict[str, str] = {
    "worldbank":   (
        "25 World Bank indicators: GDP, inflation, GNI (PPP), gini, poverty, life expectancy, "
        "infant mortality, health & education spending, physicians, sanitation, clean water, "
        "electricity access, internet usage, urban population, CO2 emissions, "
        "female labour participation, women in parliament, homicide rate, "
        "rule of law, control of corruption, political stability"
    ),
    "employment":  (
        "ILO + World Bank labour data: unemployment rate, youth unemployment, "
        "labour force participation (total & female)"
    ),
    "environment": (
        "Open-Meteo climate archive (temperature anomaly, precipitation) + "
        "World Bank / OpenAQ air quality (PM2.5) — covers noise/pollution proxies"
    ),
    "acled":       (
        "ACLED armed conflict events, fatalities, event types — "
        "covers safety, crime, conflict, and displacement pressure"
    ),
    "unhcr":       (
        "UNHCR Population API: refugee displacement outflows by year — "
        "covers forced migration, asylum seekers, stateless persons"
    ),
    "teleport":    (
        "Quality-of-life composite scores derived from World Bank data: "
        "Healthcare, Education, Safety, Economy, Cost of Living (each 0–10)"
    ),
    "news":        (
        "NewsAPI + GDELT: recent news headlines and sentiment score — "
        "covers current events, policy changes, social issues"
    ),
}

# ---------------------------------------------------------------------------
# Maps each tool name → dataset key(s) it populates in MigrationDataset.
# Used to compute per-tool row counts (tool_stats) after collection.
# ---------------------------------------------------------------------------

TOOL_DATA_KEYS: dict[str, list[str]] = {
    "worldbank":   ["worldbank"],
    "employment":  ["employment"],
    "environment": ["climate", "aqi"],
    "acled":       ["conflict_events"],
    "unhcr":       ["displacement"],
    "teleport":    ["city_scores"],
    "news":        ["news"],
}
