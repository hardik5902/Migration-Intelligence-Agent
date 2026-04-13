"""Simplified country comparison pipeline.

Flow:
  1. LLM analyses the user query → selects tools + K countries (ToolSelectorOutput)
  2. Selected tools fetch data for every country in parallel
  3. 4 comparison charts are built from the collected data
  4. LLM reads a compact data summary → returns 3 evidence-backed insights
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from google import genai
from google.genai import types

from agents.progress_tracker import reset_tracker
from agents.scout_service import collect_migration_dataset
from analysis.country_charts import build_country_comparison_charts
from models.schemas import (
    CountryComparisonResult,
    Evidence,
    IntentConfig,
    ToolSelectorOutput,
)
from tools.country_codes import country_name_to_iso3

# ---------------------------------------------------------------------------
# Available tools (shown in UI and given to the LLM)
# ---------------------------------------------------------------------------

AVAILABLE_TOOLS: dict[str, str] = {
    "worldbank": "GDP per capita, inflation, health & education spending, poverty",
    "unhcr": "Refugee displacement outflows and destination countries",
    "acled": "Armed conflict events and fatalities",
    "teleport": "City quality-of-life scores (housing, safety, education, healthcare)",
    "news": "News coverage and GDELT media sentiment over time",
    "climate": "Temperature anomalies and annual precipitation trends",
    "employment": "Unemployment rates, youth unemployment, labour participation",
    "aqi": "Air quality PM2.5 measurements by city",
}

# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

TOOL_SELECTOR_INSTRUCTION = """You are a migration intelligence tool selector.

Given a user query, decide:
1. Which data tools to use (choose 2-5 most relevant from the list below).
2. Which K countries to compare (extract K from the query; default 5).
3. A suitable year range (default 2015-2023).

Available tools:
- worldbank  : GDP per capita, inflation, health spending, education spending, poverty
- unhcr      : Refugee displacement outflows and top destination countries
- acled      : Armed conflict events and fatalities
- teleport   : City quality-of-life scores (housing, safety, education, healthcare)
- news       : News articles and GDELT media sentiment timeline
- climate    : Temperature anomalies and annual precipitation data
- employment : Unemployment rates, youth unemployment, labour force participation
- aqi        : Air quality PM2.5 measurements by city

Selection rules:
- Always include worldbank for economic / relocation queries.
- For migration push-factor queries include unhcr + worldbank.
- For safety queries include acled and/or teleport.
- For environment / health queries include aqi and/or climate.
- For economic queries include worldbank and employment.
- Identify K distinct countries that best answer the query.
  - If specific countries are named, include them.
  - If K is not specified, default to 5.
  - For "best countries to migrate to" queries pick top global destinations.
- year_from / year_to should give meaningful historical context.

Return valid JSON with EXACTLY these fields (no extra keys):
{
  "selected_tools": ["worldbank", "employment"],
  "countries": ["Germany", "Canada", "Australia", "Netherlands", "Sweden"],
  "country_codes": ["DEU", "CAN", "AUS", "NLD", "SWE"],
  "k": 5,
  "query_focus": "Compare economic stability and employment for migration decisions",
  "year_from": 2015,
  "year_to": 2023,
  "reasoning": "Selected worldbank and employment to compare economic conditions…"
}"""

EVIDENCE_GENERATOR_INSTRUCTION = """You are a migration intelligence analyst generating key insights.

Given a user query and comparative country data, generate EXACTLY 3 evidence-backed insights.

Rules:
- Each insight must cite a specific number or metric present in the data summary.
- The 3 insights should cover different dimensions (e.g. economic, safety, environment).
- Do NOT fabricate numbers — only use values in the data summary.
- title: 5-10 words, specific to the finding.
- description: 2-3 sentences explaining migration significance.

Return EXACTLY this JSON array (no extra wrapping, no markdown):
[
  {
    "title": "Germany Leads in Economic Stability",
    "value": "GDP per capita $48,718 (2022)",
    "data_source": "World Bank",
    "description": "Germany shows the strongest economic fundamentals …"
  },
  { … },
  { … }
]"""

# ---------------------------------------------------------------------------
# Genai client helper
# ---------------------------------------------------------------------------

def _get_client() -> genai.Client:
    if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI") == "TRUE":
        return genai.Client(
            vertexai=True,
            project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )
    return genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))


# ---------------------------------------------------------------------------
# Step 1 — LLM selects tools + countries
# ---------------------------------------------------------------------------

async def analyze_query_with_llm(query: str) -> ToolSelectorOutput:
    client = _get_client()
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

        # Ensure country codes list matches countries list length
        if len(result.country_codes) != len(result.countries):
            result.country_codes = [
                country_name_to_iso3(c) or c[:3].upper()
                for c in result.countries
            ]

        return result

    except Exception as exc:
        print(f"[PIPELINE] Tool selector LLM failed: {exc}. Using fallback.")
        countries = ["Germany", "Canada", "Australia", "Netherlands", "Sweden"]
        return ToolSelectorOutput(
            selected_tools=["worldbank", "employment", "aqi"],
            countries=countries,
            country_codes=[country_name_to_iso3(c) or c[:3].upper() for c in countries],
            k=5,
            query_focus="General country comparison",
            year_from=2015,
            year_to=2023,
            reasoning=f"Fallback due to error: {exc}",
        )


# ---------------------------------------------------------------------------
# Step 2 — Collect data for every country in parallel
# ---------------------------------------------------------------------------

async def collect_data_for_countries(
    countries: list[str],
    country_codes: list[str],
    selected_tools: list[str],
    year_from: int,
    year_to: int,
) -> dict[str, Any]:
    tool_flags = {t: (t in selected_tools) for t in AVAILABLE_TOOLS}

    async def fetch_one(country: str, code: str) -> tuple[str, Any]:
        intent = IntentConfig(
            intent="push_factor",
            country=country,
            country_code=code,
            year_from=year_from,
            year_to=year_to,
        )
        try:
            dataset = await collect_migration_dataset(intent, tool_flags)
            return country, dataset.model_dump()
        except Exception as exc:
            print(f"[PIPELINE] Data collection failed for {country}: {exc}")
            empty: dict[str, Any] = {
                "error": str(exc),
                "country": country,
                "worldbank": [],
                "employment": [],
                "aqi": [],
                "climate": [],
                "displacement": [],
                "conflict_events": [],
                "city_scores": [],
            }
            return country, empty

    tasks = [fetch_one(c, code) for c, code in zip(countries, country_codes)]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    out: dict[str, Any] = {}
    for item in raw_results:
        if isinstance(item, Exception):
            print(f"[PIPELINE] gather exception: {item}")
            continue
        country, data = item
        out[country] = data

    return out


# ---------------------------------------------------------------------------
# Step 4 — LLM generates exactly 3 evidence insights
# ---------------------------------------------------------------------------

def _build_data_summary(
    countries_data: dict[str, Any],
    selected_tools: list[str],
) -> str:
    import pandas as pd  # local import to keep top-level clean

    parts: list[str] = []
    for country, dataset in countries_data.items():
        lines = [f"=== {country} ==="]

        if "worldbank" in selected_tools:
            wb = pd.DataFrame(dataset.get("worldbank") or [])
            if not wb.empty and "label" in wb.columns and "value" in wb.columns:
                for label, display in [
                    ("gdp_per_capita_usd", "GDP per capita (USD)"),
                    ("inflation", "Inflation (%)"),
                    ("health_expenditure_gdp", "Health expenditure (% GDP)"),
                    ("education_spend_gdp", "Education spending (% GDP)"),
                    ("poverty_headcount", "Poverty headcount (%)"),
                ]:
                    sub = wb[wb["label"] == label].dropna(subset=["value", "year"])
                    if not sub.empty:
                        row = sub.sort_values("year").iloc[-1]
                        lines.append(
                            f"  {display}: {float(row['value']):.1f} ({int(row.get('year', 0))})"
                        )

        if "employment" in selected_tools:
            emp = pd.DataFrame(dataset.get("employment") or [])
            if not emp.empty and "unemployment_rate" in emp.columns:
                sub = emp.dropna(subset=["unemployment_rate", "year"])
                if not sub.empty:
                    row = sub.sort_values("year").iloc[-1]
                    lines.append(
                        f"  Unemployment rate: {float(row['unemployment_rate']):.1f}% ({int(row.get('year', 0))})"
                    )
                    if "youth_unemployment_rate" in emp.columns:
                        ysub = emp.dropna(subset=["youth_unemployment_rate"])
                        if not ysub.empty:
                            yrow = ysub.sort_values("year").iloc[-1]
                            lines.append(
                                f"  Youth unemployment: {float(yrow['youth_unemployment_rate']):.1f}%"
                            )

        if "aqi" in selected_tools:
            aqi = pd.DataFrame(dataset.get("aqi") or [])
            if not aqi.empty and "pm25" in aqi.columns:
                avg = aqi["pm25"].dropna().mean()
                if pd.notna(avg):
                    lines.append(f"  Avg PM2.5: {float(avg):.1f} μg/m³")

        if "climate" in selected_tools:
            clim = pd.DataFrame(dataset.get("climate") or [])
            if not clim.empty and "avg_temp_anomaly_c" in clim.columns:
                avg = clim["avg_temp_anomaly_c"].dropna().mean()
                if pd.notna(avg):
                    lines.append(f"  Avg temp anomaly: {float(avg):.2f}°C")

        if "unhcr" in selected_tools:
            disp = pd.DataFrame(dataset.get("displacement") or [])
            if not disp.empty and "value" in disp.columns:
                total = disp["value"].sum()
                lines.append(f"  Total displacement outflow: {total:,.0f} persons")

        if "acled" in selected_tools:
            conf = pd.DataFrame(dataset.get("conflict_events") or [])
            if not conf.empty:
                lines.append(f"  Conflict events: {len(conf)}")
                if "fatalities" in conf.columns:
                    lines.append(f"  Total fatalities: {int(conf['fatalities'].sum()):,}")

        if "teleport" in selected_tools:
            cs = pd.DataFrame(dataset.get("city_scores") or [])
            if not cs.empty and "score_out_of_10" in cs.columns:
                avg = cs["score_out_of_10"].dropna().mean()
                if pd.notna(avg):
                    lines.append(f"  Quality-of-life score: {float(avg):.1f}/10")

        if len(lines) > 1:
            parts.append("\n".join(lines))

    return "\n\n".join(parts) if parts else "No data available."


async def generate_evidences(
    query: str,
    countries_data: dict[str, Any],
    selected_tools: list[str],
) -> list[Evidence]:
    client = _get_client()
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    data_summary = _build_data_summary(countries_data, selected_tools)
    countries = list(countries_data.keys())

    prompt = (
        f"Query: {query}\n\n"
        f"Countries analysed: {', '.join(countries)}\n"
        f"Tools used: {', '.join(selected_tools)}\n\n"
        f"Data summary:\n{data_summary}\n\n"
        "Generate exactly 3 key evidence-backed insights from this data."
    )

    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=EVIDENCE_GENERATOR_INSTRUCTION,
                response_mime_type="application/json",
                temperature=0.2,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        raw = json.loads(response.text)
        # Accept both a bare list and {"evidences": [...]}
        if isinstance(raw, dict):
            raw = raw.get("evidences") or list(raw.values())[0]
        evidences = [Evidence.model_validate(e) for e in (raw or [])[:3]]
    except Exception as exc:
        print(f"[PIPELINE] Evidence generation failed: {exc}")
        evidences = []

    # Pad to exactly 3 if LLM returned fewer
    while len(evidences) < 3:
        evidences.append(
            Evidence(
                title="Limited Data Available",
                value="Partial dataset",
                data_source=selected_tools[0] if selected_tools else "N/A",
                description=(
                    "Insufficient data was returned for this dimension. "
                    "Consider expanding the year range or selecting additional tools."
                ),
            )
        )

    return evidences[:3]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_country_pipeline(query: str) -> CountryComparisonResult:
    """Execute the full 4-step country comparison pipeline."""
    reset_tracker()

    # Step 1: LLM selects tools + countries
    print(f"\n[PIPELINE] Step 1 — analysing query: {query!r}")
    selector = await analyze_query_with_llm(query)
    print(f"[PIPELINE]   tools   : {selector.selected_tools}")
    print(f"[PIPELINE]   countries: {selector.countries}")
    print(f"[PIPELINE]   years   : {selector.year_from}–{selector.year_to}")

    # Step 2: Collect data for each country in parallel
    print(f"\n[PIPELINE] Step 2 — collecting data for {len(selector.countries)} countries …")
    countries_data = await collect_data_for_countries(
        selector.countries,
        selector.country_codes,
        selector.selected_tools,
        selector.year_from,
        selector.year_to,
    )

    # Steps 3 + 4 run in parallel: chart building (CPU/sync) and evidence LLM call (I/O)
    print("\n[PIPELINE] Steps 3+4 — building charts and generating evidence in parallel …")
    chart_jsons, evidences = await asyncio.gather(
        asyncio.to_thread(
            build_country_comparison_charts,
            countries_data,
            selector.selected_tools,
            selector.query_focus,
        ),
        generate_evidences(query, countries_data, selector.selected_tools),
    )

    return CountryComparisonResult(
        query=query,
        countries=selector.countries,
        country_codes=selector.country_codes,
        tools_used=selector.selected_tools,
        evidences=evidences,
        summary=selector.reasoning,
        chart_jsons=chart_jsons,
        query_focus=selector.query_focus,
        year_from=selector.year_from,
        year_to=selector.year_to,
    )
