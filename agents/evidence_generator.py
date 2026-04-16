"""Step 4 of the country pipeline: hypothesis generation grounded in EDA findings.

Produces exactly 3 HypothesisInsight objects.  Each must:
  - State a clear hypothesis derived from collected data (not model weights)
  - Cite a specific number (value + year) as the anchor metric
  - List evidence_for (2-3 data points supporting the claim)
  - List evidence_against (1-2 data points that complicate or challenge it)
  - Identify a competing hypothesis and explain why the primary one wins
  - Provide explicit reasoning from data → conclusion
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd
from google.genai import types

from agents.tool_selector import get_genai_client
from models.schemas import HypothesisInsight

# ---------------------------------------------------------------------------
# LLM system prompt
# ---------------------------------------------------------------------------

HYPOTHESIS_INSTRUCTION = """You are a migration intelligence analyst. Your task is to form hypotheses \
that are DERIVED FROM DATA — not from general knowledge.

Given a user query, EDA statistical findings, and a raw data summary, produce EXACTLY 3 hypothesis insights.

CRITICAL RULES:
1. Every claim MUST cite a specific number from the data summary or EDA findings.
   Bad: "Germany has strong economic growth"
   Good: "Germany's GDP growth averaged 1.8% (2022), the highest in the sample"
2. summary: EXACTLY 1 sentence with 1-2 specific numbers. MAX 25 words.
3. evidence_for: each item MUST be in format "Country/metric: value (year)" — max 60 chars. No full sentences.
4. evidence_against: each item in format "Caveat: value/note" — max 60 chars. No full sentences.
5. Do NOT include a "reasoning" field — omit it entirely.
6. competing_hypothesis: one compact sentence naming the alternative explanation.
7. competing_verdict: one short sentence (max 15 words) on why the primary wins.
8. FOCUS on the query topic specified in ANALYSIS FOCUS. Do NOT hypothesize about
   unrelated metrics (e.g. if query is about internet access, do not write about
   political stability or GDP unless directly relevant).
9. Each of the 3 insights must draw on a DIFFERENT aspect of the query topic.
10. confidence scoring:
    - 85-100: multi-year trend, primary source, ≥ 4 countries have the data
    - 65-84:  1-2 year data or partial country coverage
    - 45-64:  proxy/fallback data, limited sample
    - < 45:   very sparse data, wide uncertainty

Return EXACTLY this JSON array (no markdown, no extra keys):
[
  {
    "headline": "South Korea leads internet penetration — top destination for digital workers",
    "summary": "South Korea reaches 97.6% internet users (2023), 18pp above the sample average.",
    "key_metric": "Internet users 97.6% (2023)",
    "evidence_for": [
      "South Korea: 97.6% internet users (2023)",
      "Electricity access: 100% (2022)",
      "CAGR +0.4%/yr — sustained growth"
    ],
    "evidence_against": [
      "Caveat: data plateaus near 100% — growth potential limited",
      "Caveat: urban-rural gap may understate rural access"
    ],
    "competing_hypothesis": "Japan matches on infrastructure quality despite lower headline internet %.",
    "competing_verdict": "Korea's CAGR trend is more consistent than Japan's flat trajectory.",
    "data_source": "World Bank",
    "confidence": 88
  }
]"""

# ---------------------------------------------------------------------------
# Detailed EDA summary builder (feeds into LLM prompt)
# ---------------------------------------------------------------------------

def _build_eda_summary(eda_findings: dict[str, Any]) -> str:
    """Convert rich EDA findings into a precise, number-dense prompt block."""
    if not eda_findings:
        return ""

    lines = ["=== EDA Statistical Findings (derived from data, not model weights) ==="]

    # --- Narrative findings ---
    for f in eda_findings.get("findings", []):
        tag = f.get("type", "").upper()
        lines.append(f"\n[{tag}] {f.get('title', '')}")
        lines.append(f"  Key value: {f.get('value', '')}")
        if f.get("detail"):
            lines.append(f"  Detail: {f['detail']}")

    # --- CAGR growth rates per country ---
    growth_rates = eda_findings.get("growth_rates", {})
    if growth_rates:
        lines.append("\n--- CAGR growth rates per country ---")
        for country, metrics in growth_rates.items():
            for metric, gr in metrics.items():
                cagr = gr.get("cagr")
                if cagr is not None:
                    peak = gr.get("peak_growth_year")
                    trough = gr.get("trough_year")
                    lines.append(
                        f"  {country} {metric}: CAGR {cagr * 100:+.2f}%"
                        + (f"  peak {peak}" if peak else "")
                        + (f"  trough {trough}" if trough else "")
                    )

    # --- Statistical summary (mean ± std, min/max) ---
    stats_summary = eda_findings.get("stats_summary", {})
    if stats_summary:
        lines.append("\n--- Statistical summary (mean ± std) ---")
        for country, metrics in stats_summary.items():
            parts = []
            for metric, s in metrics.items():
                if s.get("mean") is not None:
                    parts.append(
                        f"{metric}: μ={s['mean']:.2f} σ={s['std']:.2f} "
                        f"[{s['min']:.1f}–{s['max']:.1f}] n={s['n']}"
                    )
            if parts:
                lines.append(f"  {country}: " + " | ".join(parts))

    # --- Anomalies ---
    anomalies = eda_findings.get("anomalies", {})
    any_anomalies = {c: a for c, a in anomalies.items() if a}
    if any_anomalies:
        lines.append("\n--- Statistical anomalies (|z| > 2) ---")
        for country, anom_list in any_anomalies.items():
            for a in anom_list:
                years_str = ", ".join(str(y) for y in a["years"])
                z_str = ", ".join(f"{z:.1f}" for z in a.get("z_scores", []))
                lines.append(
                    f"  {country} {a['metric']}: outlier year(s) {years_str}"
                    + (f" (z={z_str})" if z_str else "")
                )

    # --- Top correlations ---
    correlations = eda_findings.get("correlations", [])
    if correlations:
        lines.append("\n--- Cross-country correlations (Pearson r, target = GDP growth) ---")
        for c in correlations[:5]:
            sig = "significant" if c.get("pearson_p", 1) < 0.05 else "not significant"
            lines.append(
                f"  {c.get('indicator','')} ↔ GDP growth: "
                f"r={c.get('pearson_r',0):+.3f}  p={c.get('pearson_p',1):.3f}  ({sig})"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Raw data summary builder
# ---------------------------------------------------------------------------

def _build_data_summary(
    countries_data: dict[str, Any],
    selected_tools: list[str],
) -> str:
    """Convert collected datasets into a compact, number-dense text block."""
    parts: list[str] = []

    for country, dataset in countries_data.items():
        lines = [f"=== {country} ==="]

        if "worldbank" in selected_tools:
            wb = pd.DataFrame(dataset.get("worldbank") or [])
            if not wb.empty and "label" in wb.columns and "value" in wb.columns:
                for label, display in [
                    # Economy
                    ("gdp_growth",              "GDP growth (%)"),
                    ("gdp_per_capita_usd",      "GDP per capita (USD)"),
                    ("gni_per_capita_ppp",       "GNI per capita PPP (int. $)"),
                    ("inflation",               "Inflation (%)"),
                    ("gini",                    "Gini index"),
                    ("poverty_headcount",       "Poverty headcount (%)"),
                    # Governance
                    ("political_stability",     "Political stability index"),
                    ("control_of_corruption",   "Control of corruption index"),
                    ("rule_of_law",             "Rule of law index"),
                    ("homicide_rate",           "Homicide rate (per 100k)"),
                    # Health
                    ("health_expenditure_gdp",  "Health expenditure (% GDP)"),
                    ("physicians_per_1000",     "Physicians per 1,000 people"),
                    ("life_expectancy",         "Life expectancy (years)"),
                    ("infant_mortality",        "Infant mortality (per 1,000 births)"),
                    # Education
                    ("education_spend_gdp",     "Education spending (% GDP)"),
                    # Infrastructure
                    ("sanitation_access",       "Sanitation access (% pop.)"),
                    ("clean_water_access",      "Clean water access (% pop.)"),
                    ("electricity_access",      "Electricity access (% pop.)"),
                    ("internet_users_pct",      "Internet users (% pop.)"),
                    ("urban_population_pct",    "Urban population (% total)"),
                    # Gender
                    ("female_labor_participation", "Female labour participation (%)"),
                    ("women_in_parliament",     "Women in parliament (%)"),
                    ("adolescent_fertility_rate", "Adolescent fertility rate"),
                    # Environment
                    ("co2_per_capita",          "CO₂ per capita (t)"),
                ]:
                    sub = wb[wb["label"] == label].dropna(subset=["value", "year"])
                    if not sub.empty:
                        row = sub.sort_values("year").iloc[-1]
                        lines.append(
                            f"  {display}: {float(row['value']):.2f} ({int(row.get('year', 0))})"
                        )
                        # Also show earliest value for trend context
                        if len(sub) > 1:
                            first_row = sub.sort_values("year").iloc[0]
                            lines.append(
                                f"    earliest: {float(first_row['value']):.2f} "
                                f"({int(first_row.get('year', 0))})"
                            )

        if "employment" in selected_tools:
            emp = pd.DataFrame(dataset.get("employment") or [])
            if not emp.empty:
                for col, display in [
                    ("unemployment_rate",       "Unemployment rate (%)"),
                    ("youth_unemployment_rate", "Youth unemployment (%)"),
                    ("labor_force_participation","Labour force participation (%)"),
                ]:
                    if col in emp.columns:
                        sub = emp.dropna(subset=[col, "year"])
                        if not sub.empty:
                            row = sub.sort_values("year").iloc[-1]
                            lines.append(
                                f"  {display}: {float(row[col]):.2f}%"
                                f" ({int(row.get('year', 0))})"
                            )

        if "teleport" in selected_tools or True:  # teleport is always collected
            cs = pd.DataFrame(dataset.get("city_scores") or [])
            if not cs.empty and "score_out_of_10" in cs.columns:
                avg_score = cs["score_out_of_10"].dropna().mean()
                if pd.notna(avg_score):
                    lines.append(f"  Quality-of-life score (Teleport): {float(avg_score):.2f}/10")
                # Show per-category breakdown if available
                if "category" in cs.columns:
                    for _, cat_row in cs.dropna(subset=["category", "score_out_of_10"]).iterrows():
                        lines.append(
                            f"    {cat_row['category']}: {float(cat_row['score_out_of_10']):.2f}/10"
                        )

        if "unhcr" in selected_tools:
            disp = pd.DataFrame(dataset.get("displacement") or [])
            if not disp.empty and "value" in disp.columns:
                total = disp["value"].sum()
                if total > 0:
                    lines.append(f"  UNHCR displaced persons (total): {int(total):,}")
                    # Year breakdown for trend context
                    if "year" in disp.columns:
                        yearly = (
                            disp.dropna(subset=["year", "value"])
                            .groupby("year")["value"].sum()
                            .sort_index()
                        )
                        for yr, val in yearly.tail(5).items():
                            lines.append(f"    {int(yr)}: {int(val):,}")

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
                lines.append(f"  Recent news ({len(headlines)} articles):")
                for h in headlines[:5]:
                    lines.append(f"    - {h}")

        if len(lines) > 1:
            parts.append("\n".join(lines))

    return "\n\n".join(parts) if parts else "No data available."


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

async def generate_hypotheses(
    query: str,
    countries_data: dict[str, Any],
    selected_tools: list[str],
    eda_findings: dict[str, Any] | None = None,
    query_focus: str = "",
) -> list[HypothesisInsight]:
    """
    Ask the LLM to produce exactly 3 hypothesis insights grounded in EDA data.

    Returns a list of HypothesisInsight objects with evidence-for, evidence-against,
    and competing hypotheses — meeting the professor's requirements for Step 3.
    """
    client = get_genai_client()
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")

    data_summary = _build_data_summary(countries_data, selected_tools)
    eda_summary  = _build_eda_summary(eda_findings or {})
    countries    = list(countries_data.keys())
    focus        = query_focus or query or "general country comparison"

    prompt = (
        f"ANALYSIS FOCUS: {focus}\n\n"
        f"User query: {query}\n\n"
        f"Countries: {', '.join(countries)}\n"
        f"Tools used: {', '.join(selected_tools)}\n\n"
        + (f"{eda_summary}\n\n" if eda_summary else "")
        + f"Raw data summary:\n{data_summary}\n\n"
        "IMPORTANT: All 3 hypotheses MUST be directly about the query topic above. "
        "Do not hypothesize about unrelated metrics even if they appear in the data. "
        "Form exactly 3 hypotheses. Each must be derived from the data above — "
        "cite specific numbers from the EDA findings or raw data, not general knowledge."
    )

    hypotheses: list[HypothesisInsight] = []
    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=HYPOTHESIS_INSTRUCTION,
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        raw = json.loads(response.text)
        if isinstance(raw, dict):
            raw = raw.get("hypotheses") or list(raw.values())[0]
        hypotheses = [HypothesisInsight.model_validate(h) for h in (raw or [])[:3]]
    except Exception as exc:
        print(f"[HYPOTHESIS_GENERATOR] LLM failed: {exc}")

    # Pad to exactly 3 with a fallback placeholder if LLM returned fewer
    while len(hypotheses) < 3:
        hypotheses.append(HypothesisInsight(
            headline="Insufficient data for this dimension",
            summary="Insufficient data returned — expand year range or add tools.",
            key_metric="N/A",
            evidence_for=["Partial dataset returned"],
            evidence_against=["Caveat: cannot confirm without more data"],
            competing_hypothesis="",
            competing_verdict="",
            data_source=selected_tools[0] if selected_tools else "N/A",
            confidence=25,
        ))

    return hypotheses[:3]


# ---------------------------------------------------------------------------
# Backward-compatible alias kept for any callers that import generate_evidences
# ---------------------------------------------------------------------------

async def generate_evidences(
    query: str,
    countries_data: dict[str, Any],
    selected_tools: list[str],
    eda_findings: dict[str, Any] | None = None,
    query_focus: str = "",
) -> list[HypothesisInsight]:
    """Alias for generate_hypotheses — preserves existing call sites."""
    return await generate_hypotheses(query, countries_data, selected_tools, eda_findings, query_focus)
