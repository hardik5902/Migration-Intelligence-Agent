"""Shared constants for the country comparison pipeline.

Centralised here so tool_selector, data_collector, evidence_generator and the
orchestrators all reference the same source of truth.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Tool registry — shown in the UI and fed to the LLM tool selector
# ---------------------------------------------------------------------------

AVAILABLE_TOOLS: dict[str, str] = {
    "worldbank":   "GDP growth, inflation, health & education spending, poverty, political stability",
    "employment":  "Unemployment rates, youth unemployment, labour force participation (ILO + World Bank)",
    "unhcr":       "Refugee displacement outflows and destination countries",
    "environment": "Climate trends (temperature, precipitation) + air quality PM2.5 — combined",
    "acled":       "Armed conflict events and fatalities",
    "news":        "Recent news events summarised around the query topic",
}

# ---------------------------------------------------------------------------
# Maps each tool name → the dataset key(s) it populates in MigrationDataset
# Used to compute per-tool row counts (tool_stats) after collection.
# ---------------------------------------------------------------------------

TOOL_DATA_KEYS: dict[str, list[str]] = {
    "worldbank":   ["worldbank"],
    "employment":  ["employment"],
    "unhcr":       ["displacement"],
    "environment": ["climate", "aqi"],
    "acled":       ["conflict_events"],
    "news":        ["news"],
}
