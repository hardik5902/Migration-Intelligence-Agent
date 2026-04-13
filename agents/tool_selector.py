"""Step 1 of the country pipeline: LLM analyses the user query and selects
tools + K countries to compare.
"""

from __future__ import annotations

import json
import os
from typing import Any

from google import genai
from google.genai import types

from models.schemas import ToolSelectorOutput
from tools.country_codes import country_name_to_iso3

# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

TOOL_SELECTOR_INSTRUCTION = """You are a migration intelligence tool selector.

Given a user query, decide:
1. Which data tools to use (choose 2-4 most relevant from the list below).
2. Which K countries to compare (extract K from the query; default 5).
3. A suitable year range (default 2020-2024).

Available tools:
- worldbank   : GDP growth, inflation, health spending, education spending, poverty, political stability
- employment  : Unemployment rates, youth unemployment, labour force participation
- unhcr       : Refugee displacement outflows and top destination countries
- environment : Climate trends (temperature anomaly, precipitation) + PM2.5 air quality — use for health/environment queries
- acled       : Armed conflict events and fatalities — use for safety/conflict queries
- news        : Recent news summary focused on the query topic — use when current events matter

Selection rules:
- ALWAYS include worldbank — it provides the economic foundation for every query.
- For migration push-factor queries: worldbank + unhcr.
- For safety/conflict queries: worldbank + acled.
- For environment/health/climate queries: worldbank + environment.
- For economic/labour queries: worldbank + employment.
- For current-events queries: worldbank + news.
- General relocation queries: worldbank + employment + unhcr.
- Add a 3rd or 4th tool only when the query clearly calls for it.
- Identify K distinct countries that best answer the query.
  - If specific countries are named, include them.
  - If K is not specified, default to 5.
  - For "best countries to migrate to" pick top global destinations (Germany, Canada, Australia etc).
- year_from / year_to should give meaningful historical context (default 2020-2025).

Return valid JSON with EXACTLY these fields (no extra keys):
{
  "selected_tools": ["worldbank", "employment", "unhcr"],
  "countries": ["Germany", "Canada", "Australia", "Netherlands", "Sweden"],
  "country_codes": ["DEU", "CAN", "AUS", "NLD", "SWE"],
  "k": 5,
  "query_focus": "Compare economic stability and employment for migration decisions",
  "year_from": 2015,
  "year_to": 2023,
  "reasoning": "worldbank always included; added employment for labour data and unhcr for displacement flows"
}"""

# ---------------------------------------------------------------------------
# Genai client helper (shared with evidence_generator)
# ---------------------------------------------------------------------------

def get_genai_client() -> genai.Client:
    if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI") == "TRUE":
        return genai.Client(
            vertexai=True,
            project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )
    return genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

async def analyze_query_with_llm(query: str) -> ToolSelectorOutput:
    """Call the LLM to pick tools + K countries for the given query."""
    client = get_genai_client()
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    prompt = f"Query: {query}\n\nAnalyse this query and select the appropriate tools and countries."

    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=TOOL_SELECTOR_INSTRUCTION,
                response_mime_type="application/json",
                temperature=0.1,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        raw = json.loads(response.text)
        result = ToolSelectorOutput.model_validate(raw)

        # Ensure country_codes length matches countries length
        if len(result.country_codes) != len(result.countries):
            result.country_codes = [
                country_name_to_iso3(c) or c[:3].upper()
                for c in result.countries
            ]
        return result

    except Exception as exc:
        # Re-raise so the pipeline surfaces the real error to the user
        # rather than silently running with static countries unrelated to the query.
        raise RuntimeError(
            f"Tool selector LLM call failed — please retry. Original error: {exc}"
        ) from exc
