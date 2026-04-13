"""Step 4 of the country pipeline: build a compact data summary and ask the
LLM to generate exactly 3 evidence-backed insights with confidence scores.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
from google.genai import types

from agents.tool_selector import get_genai_client
from models.schemas import Evidence

# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

EVIDENCE_GENERATOR_INSTRUCTION = """You are a migration intelligence analyst generating key insights.

Given a user query and comparative country data, generate EXACTLY 3 evidence-backed insights.

CRITICAL RULES — ALL MUST BE FOLLOWED:
1. Each of the 3 insights MUST come from a DIFFERENT data source / tool.
   - insight 1 → one tool (e.g. World Bank)
   - insight 2 → a DIFFERENT tool (e.g. ILO / Employment)
   - insight 3 → yet another DIFFERENT tool (e.g. Environment, UNHCR, ACLED, News)
   - Never use the same data_source twice across the 3 insights.
2. Each insight must cite a SPECIFIC number or metric from the data summary — no invented figures.
3. If news headlines appear in the data summary, one insight MUST summarise what those events
   indicate about the query topic — use data_source "News".
4. confidence (0-100): your confidence in the insight based on:
   - 90-100: multiple years of data, primary source, consistent trend
   - 70-89:  single year or partial country coverage, still reliable
   - 50-69:  proxy/fallback data, limited sample, older year
   - below 50: very sparse data, wide uncertainty

Fields:
- title: 5-10 words, specific to the finding
- value: the key metric that anchors the insight (include units and year)
- data_source: the API/tool name (World Bank, ILO, UNHCR, ACLED, Open-Meteo, OpenAQ, News)
- description: 2-3 sentences explaining what this means for migrants
- confidence: integer 0-100

Return EXACTLY this JSON array (no extra wrapping, no markdown):
[
  {
    "title": "Germany Leads in Economic Stability",
    "value": "GDP growth 1.8% (2022)",
    "data_source": "World Bank",
    "description": "Germany shows the strongest economic fundamentals …",
    "confidence": 88
  },
  {
    "title": "Thailand Has Lower Youth Unemployment",
    "value": "4.5% vs India 15.6% (2023)",
    "data_source": "ILO",
    "description": "Youth unemployment in Thailand is far lower …",
    "confidence": 74
  },
  {
    "title": "India Faces Worsening Air Quality",
    "value": "PM2.5 avg 85 μg/m³ (2024)",
    "data_source": "OpenAQ",
    "description": "India's PM2.5 levels far exceed WHO limits …",
    "confidence": 65
  }
]"""

# ---------------------------------------------------------------------------
# Data summary builder
# ---------------------------------------------------------------------------

def _build_data_summary(
    countries_data: dict[str, Any],
    selected_tools: list[str],
) -> str:
    """Convert collected datasets into a compact text block for the LLM."""
    parts: list[str] = []

    for country, dataset in countries_data.items():
        lines = [f"=== {country} ==="]

        if "worldbank" in selected_tools:
            wb = pd.DataFrame(dataset.get("worldbank") or [])
            if not wb.empty and "label" in wb.columns and "value" in wb.columns:
                for label, display in [
                    ("gdp_growth",            "GDP growth (%)"),
                    ("gdp_per_capita_usd",    "GDP per capita (USD)"),
                    ("inflation",             "Inflation (%)"),
                    ("health_expenditure_gdp","Health expenditure (% GDP)"),
                    ("education_spend_gdp",   "Education spending (% GDP)"),
                    ("poverty_headcount",     "Poverty headcount (%)"),
                    ("political_stability",   "Political stability index"),
                    ("gini",                  "Gini index"),
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
                        f"  Unemployment rate: {float(row['unemployment_rate']):.1f}%"
                        f" ({int(row.get('year', 0))})"
                    )
                if "youth_unemployment_rate" in emp.columns:
                    ysub = emp.dropna(subset=["youth_unemployment_rate", "year"])
                    if not ysub.empty:
                        yrow = ysub.sort_values("year").iloc[-1]
                        lines.append(
                            f"  Youth unemployment: {float(yrow['youth_unemployment_rate']):.1f}%"
                            f" ({int(yrow.get('year', 0))})"
                        )

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

        if "environment" in selected_tools:
            clim = pd.DataFrame(dataset.get("climate") or [])
            if not clim.empty and "avg_temp_anomaly_c" in clim.columns:
                avg = clim["avg_temp_anomaly_c"].dropna().mean()
                if pd.notna(avg):
                    lines.append(f"  Avg temp anomaly: {float(avg):.2f}°C")
            aqi_df = pd.DataFrame(dataset.get("aqi") or [])
            if not aqi_df.empty and "pm25" in aqi_df.columns:
                valid = aqi_df["pm25"].dropna()
                valid = valid[(valid > 0) & (valid <= 1000)]
                if not valid.empty:
                    lines.append(f"  Avg PM2.5: {float(valid.mean()):.1f} μg/m³")

        if "news" in selected_tools:
            news_list = dataset.get("news") or []
            headlines = [r.get("title", "").strip() for r in news_list if r.get("title")]
            if headlines:
                lines.append(f"  Recent news headlines ({len(headlines)} articles):")
                for h in headlines[:6]:
                    lines.append(f"    - {h}")

        if len(lines) > 1:
            parts.append("\n".join(lines))

    return "\n\n".join(parts) if parts else "No data available."


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

async def generate_evidences(
    query: str,
    countries_data: dict[str, Any],
    selected_tools: list[str],
) -> list[Evidence]:
    """Ask the LLM to produce exactly 3 evidence insights from collected data."""
    client = get_genai_client()
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

    evidences: list[Evidence] = []
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
        if isinstance(raw, dict):
            raw = raw.get("evidences") or list(raw.values())[0]
        evidences = [Evidence.model_validate(e) for e in (raw or [])[:3]]
    except Exception as exc:
        print(f"[EVIDENCE_GENERATOR] LLM failed: {exc}")

    # Pad to exactly 3 if LLM returned fewer
    while len(evidences) < 3:
        evidences.append(Evidence(
            title="Limited Data Available",
            value="Partial dataset",
            data_source=selected_tools[0] if selected_tools else "N/A",
            description=(
                "Insufficient data was returned for this dimension. "
                "Consider expanding the year range or selecting additional tools."
            ),
            confidence=30,
        ))

    return evidences[:3]
