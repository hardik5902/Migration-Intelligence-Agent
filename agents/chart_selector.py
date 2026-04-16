"""LLM-based chart selection: picks the 4 most relevant charts from the registry.

Replaces the heuristic tag-scoring approach with a Gemini call that sees the
full data manifest (what data is available, for which countries) and selects
the 4 charts that best answer the user's query.
"""

from __future__ import annotations

import json
import os
from typing import Any

from google.genai import types

from agents.tool_selector import get_genai_client

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

CHART_SELECTOR_INSTRUCTION = """You are a data visualization expert for a migration intelligence platform.

You receive a user query and a list of available charts. Each chart entry has:
- key: stable identifier you must return
- title: human-readable chart name
- type: "line" | "bar" | "scatter" | "area"
- tags: topic categories this chart covers
- countries_with_data: which countries have data for this chart

Select EXACTLY 4 chart keys that best serve the user's query.

SELECTION RULES:
1. Relevance first — pick charts whose tags match the query topic.
   Examples:
   - "education" query → education_spending charts
   - "healthcare / doctors" query → life_expectancy, infant_mortality, physicians, health_expenditure charts
   - "safety / crime" query → homicide_rate, rule_of_law, conflict_events charts
   - "environment / climate" query → co2, temp_anomaly, aqi, precipitation charts
   - "gender equality" query → women_parliament, female_labor charts
   - "internet / digital" query → internet_users, electricity_access charts
   - "income / wealth" query → gdp_per_capita, gni_per_capita charts
   - "poverty / inequality" query → poverty_headcount, gini charts
2. Variety — aim for at most 2 charts of the same visual type (line/bar/scatter/area).
3. No duplicates — do not pick two charts backed by the exact same underlying metric.
4. All chosen charts must have at least 2 entries in countries_with_data.

Return ONLY a JSON array of exactly 4 key strings. No markdown, no explanation.
Example: ["education_spending_line", "education_spending_bar", "gdp_per_capita_bar", "gdp_vs_growth_scatter"]"""


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

async def select_charts_with_llm(
    query: str,
    manifest: list[dict[str, Any]],
    selected_tools: list[str],
    query_focus: str,
) -> list[str]:
    """Ask Gemini to pick the 4 best chart keys from the available manifest.

    Falls back to an empty list on any error — the caller must handle the
    fallback (heuristic sort) when an empty list is returned.
    """
    client = get_genai_client()
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")

    # Only expose charts that actually have data for ≥2 countries
    available = [m for m in manifest if len(m.get("countries_with_data", [])) >= 2]
    if not available:
        return []

    # Keep only JSON-serialisable fields — strip all callables and _metric ref
    _SEND_KEYS = {"key", "title", "type", "tags", "source", "countries_with_data",
                  "xaxis_title", "yaxis_title"}
    safe_manifest = [
        {k: v for k, v in m.items() if k in _SEND_KEYS}
        for m in available
    ]

    prompt = (
        f"User query: {query}\n"
        f"Query focus: {query_focus}\n"
        f"Tools used: {', '.join(selected_tools)}\n\n"
        f"Available charts ({len(safe_manifest)} with data):\n"
        + json.dumps(safe_manifest, indent=2)
        + "\n\nSelect exactly 4 chart keys."
    )

    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=CHART_SELECTOR_INSTRUCTION,
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        raw = json.loads(response.text)
        # Handle both bare array and wrapped dict responses
        if isinstance(raw, dict):
            raw = raw.get("keys") or raw.get("charts") or next(iter(raw.values()), [])
        if not isinstance(raw, list):
            return []

        # Validate all returned keys exist in available manifest
        valid_keys = {m["key"] for m in available}
        keys = [str(k) for k in raw if isinstance(k, str) and k in valid_keys]
        return keys[:4]

    except Exception as exc:
        print(f"[CHART_SELECTOR] LLM failed: {exc}")
        return []
