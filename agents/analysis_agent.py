"""Unified analysis agent: single LLM call that replaces chart_selector + evidence_generator.

Given collected data, a data-coverage introspection report, the comparison chart
manifest, the EDA chart manifest, and EDA statistical findings, it returns:

  - comparison_chart_keys : list of 4 keys from the country_charts registry
  - eda_chart_keys        : list of up to 4 keys from the EDA chart manifest
  - hypotheses            : list of 3 HypothesisInsight objects

This replaces two prior LLM calls (select_charts_with_llm + generate_hypotheses)
with a single request so the model sees the full analytical context at once.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
from google.genai import types

from agents.tool_selector import get_genai_client
from models.schemas import HypothesisInsight


# ---------------------------------------------------------------------------
# Data-coverage introspection
# ---------------------------------------------------------------------------

_WB_LABELS_TO_REPORT = [
    "gdp_growth", "gdp_per_capita_usd", "gni_per_capita_ppp", "inflation",
    "gini", "poverty_headcount",
    "political_stability", "control_of_corruption", "rule_of_law", "homicide_rate",
    "health_expenditure_gdp", "physicians_per_1000", "life_expectancy", "infant_mortality",
    "education_spend_gdp",
    "sanitation_access", "clean_water_access", "electricity_access",
    "internet_users_pct", "urban_population_pct",
    "female_labor_participation", "women_in_parliament", "adolescent_fertility_rate",
    "co2_per_capita",
]


def _year_range(years: list[int]) -> str:
    if not years:
        return ""
    return f"{min(years)}-{max(years)}" if min(years) != max(years) else str(min(years))


def build_data_coverage_report(
    countries_data: dict[str, Any],
    selected_tools: list[str],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Decode what data columns actually came back per country per tool.

    Shape:
      {
        country: {
          tool: [ {"column": str, "years": "2015-2023", "n": 9}, ... ],
          ...
        },
        ...
      }
    """
    report: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for country, dataset in countries_data.items():
        per_country: dict[str, list[dict[str, Any]]] = {}

        if "worldbank" in selected_tools:
            wb = pd.DataFrame(dataset.get("worldbank") or [])
            if not wb.empty and "label" in wb.columns and "value" in wb.columns:
                cols: list[dict[str, Any]] = []
                for label in _WB_LABELS_TO_REPORT:
                    sub = wb[wb["label"] == label].dropna(subset=["value"])
                    if sub.empty:
                        continue
                    years = sub["year"].dropna().astype(int).tolist() if "year" in sub.columns else []
                    cols.append({
                        "column": label,
                        "years":  _year_range(years),
                        "n":      int(len(sub)),
                    })
                if cols:
                    per_country["worldbank"] = cols

        if "employment" in selected_tools:
            emp = pd.DataFrame(dataset.get("employment") or [])
            if not emp.empty:
                cols = []
                for col in ["unemployment_rate", "youth_unemployment_rate", "labor_force_participation"]:
                    if col in emp.columns:
                        sub = emp.dropna(subset=[col])
                        if not sub.empty:
                            years = sub["year"].dropna().astype(int).tolist() if "year" in sub.columns else []
                            cols.append({
                                "column": col,
                                "years":  _year_range(years),
                                "n":      int(len(sub)),
                            })
                if cols:
                    per_country["employment"] = cols

        if "acled" in selected_tools:
            conf = pd.DataFrame(dataset.get("conflict_events") or [])
            if not conf.empty:
                cols = [{"column": "conflict_events", "years": "", "n": int(len(conf))}]
                if "fatalities" in conf.columns:
                    cols.append({
                        "column": "fatalities",
                        "years":  "",
                        "n":      int(conf["fatalities"].dropna().shape[0]),
                    })
                per_country["acled"] = cols

        if "environment" in selected_tools:
            cols = []
            clim = pd.DataFrame(dataset.get("climate") or [])
            if not clim.empty:
                for col in ["avg_temp_anomaly_c", "annual_precipitation_mm"]:
                    if col in clim.columns:
                        sub = clim.dropna(subset=[col])
                        if not sub.empty:
                            years = sub["year"].dropna().astype(int).tolist() if "year" in sub.columns else []
                            cols.append({
                                "column": col,
                                "years":  _year_range(years),
                                "n":      int(len(sub)),
                            })
            aqi = pd.DataFrame(dataset.get("aqi") or [])
            if not aqi.empty and "pm25" in aqi.columns:
                cols.append({
                    "column": "pm25",
                    "years":  "",
                    "n":      int(aqi["pm25"].dropna().shape[0]),
                })
            if cols:
                per_country["environment"] = cols

        if "teleport" in selected_tools or True:
            cs = pd.DataFrame(dataset.get("city_scores") or [])
            if not cs.empty and "score_out_of_10" in cs.columns:
                per_country["teleport"] = [{
                    "column": "quality_of_life_score",
                    "years":  "",
                    "n":      int(cs["score_out_of_10"].dropna().shape[0]),
                }]

        if "news" in selected_tools:
            news = dataset.get("news") or []
            if news:
                per_country["news"] = [{
                    "column": "headlines",
                    "years":  "",
                    "n":      int(len(news)),
                }]

        if per_country:
            report[country] = per_country

    return report


# ---------------------------------------------------------------------------
# EDA summary builder (compact, number-dense)
# ---------------------------------------------------------------------------

def _build_eda_summary(eda_findings: dict[str, Any]) -> str:
    if not eda_findings:
        return ""

    lines = ["=== EDA Statistical Findings ==="]

    for f in eda_findings.get("findings", []):
        tag = (f.get("type") or "").upper()
        lines.append(f"[{tag}] {f.get('title', '')} — {f.get('value', '')}")
        if f.get("detail"):
            lines.append(f"  {f['detail']}")

    growth_rates = eda_findings.get("growth_rates", {})
    if growth_rates:
        lines.append("\n-- CAGR per country --")
        for country, metrics in growth_rates.items():
            for metric, gr in metrics.items():
                cagr = gr.get("cagr")
                if cagr is not None:
                    lines.append(f"  {country} {metric}: CAGR {cagr * 100:+.2f}%")

    stats_summary = eda_findings.get("stats_summary", {})
    if stats_summary:
        lines.append("\n-- Statistical summary (mean ± σ) --")
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

    anomalies = eda_findings.get("anomalies", {})
    any_anom = {c: a for c, a in anomalies.items() if a}
    if any_anom:
        lines.append("\n-- Anomalies (|z| > 2) --")
        for country, anom_list in any_anom.items():
            for a in anom_list:
                years_str = ", ".join(str(y) for y in a["years"])
                lines.append(f"  {country} {a['metric']}: outlier year(s) {years_str}")

    correlations = eda_findings.get("correlations", [])
    if correlations:
        lines.append("\n-- Top cross-country correlations (Pearson r) --")
        for c in correlations[:5]:
            sig = "significant" if c.get("pearson_p", 1) < 0.05 else "not significant"
            lines.append(
                f"  {c.get('indicator','')}: r={c.get('pearson_r',0):+.3f} "
                f"p={c.get('pearson_p',1):.3f} ({sig})"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Unified LLM system instruction
# ---------------------------------------------------------------------------

UNIFIED_ANALYSIS_INSTRUCTION = """You are the migration-intelligence analysis agent.

You receive a user query plus four pieces of context:
  1. DATA COVERAGE — what columns each tool returned for each country
  2. COMPARISON CHART MANIFEST — available country-comparison charts (with tags + country coverage)
  3. EDA CHART MANIFEST — available statistical charts (heatmap / CAGR bar / anomaly timeline / distribution box)
  4. EDA FINDINGS — pre-computed statistical findings + CAGR / anomalies / stats / correlations

Your job is to return ONE JSON object containing THREE things at once.

═══ HARD RULES (violating these makes the output invalid) ═══
1. NO REPEATED METRICS across comparison_chart_keys AND eda_chart_keys.
   If life_expectancy appears as an EDA chart, do NOT pick a comparison chart
   about life_expectancy. The two sections must show DIFFERENT metrics.
2. NO BLANK CHARTS. Only pick keys that appear in the manifest you were given.
   If a key has 0 entries in countries_with_data, skip it.
3. ONLY pick keys that are literally present in the manifests shown to you.
   Do not invent keys.
4. QUERY RELEVANCE. Every chart must directly address the user's query topic.
   Do NOT pick GDP/inflation/political_stability for a healthcare query.
   Do NOT pick health charts for an employment query.
═══════════════════════════════════════════════════════════════

A. comparison_chart_keys — UP TO 4 keys from the comparison chart manifest.
   - Return fewer than 4 if fewer have relevant data (never pad with off-topic charts).
   - Relevance first: tags must match the query topic.
   - Visual variety: at most 2 of the same chart type (line / bar / scatter / area).
   - No two keys may back the same underlying metric.
   - Each key must have ≥2 countries in countries_with_data.

B. eda_chart_keys — UP TO 4 keys from the EDA chart manifest.
   - Pick fewer than 4 if the data doesn't support more — empty list is valid.
   - Each EDA chart must cover a DIFFERENT metric from every comparison chart.
   - Each EDA chart must cover a DIFFERENT metric from every other EDA chart.
   - At most 1 anomaly_timeline, at most 1 distribution_box for any given metric.
   - Prefer: statistical insights (heatmap, CAGR growth, anomaly detection).

C. hypotheses — EXACTLY 3 HypothesisInsight objects. Rules:
   - Each claim MUST cite a specific number from EDA findings or raw coverage.
   - summary: 1 sentence, 1-2 specific numbers, ≤25 words.
   - evidence_for items: "Country/metric: value (year)" format, ≤60 chars each. No prose.
   - evidence_against items: "Caveat: value/note", ≤60 chars each. No prose.
   - Do NOT include a "reasoning" field.
   - competing_hypothesis: one compact sentence.
   - competing_verdict: ≤15 words on why the primary wins.
   - FOCUS on the query topic — do not drift to unrelated metrics.
   - Each of the 3 hypotheses must draw on a DIFFERENT aspect of the topic.
   - confidence: 85-100 multi-year trend, 65-84 partial, 45-64 proxy, <45 very sparse.

Return ONLY this JSON (no markdown, no extra keys):
{
  "comparison_chart_keys": ["key1", "key2", "key3", "key4"],
  "eda_chart_keys": ["eda_key1", "eda_key2", ...],
  "hypotheses": [
    {
      "headline": "...",
      "summary": "...",
      "key_metric": "...",
      "evidence_for": ["...", "..."],
      "evidence_against": ["...", "..."],
      "competing_hypothesis": "...",
      "competing_verdict": "...",
      "data_source": "...",
      "confidence": 85
    },
    ...
  ]
}"""


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

# Keep only JSON-safe fields from each comparison-chart manifest entry
_COMPARISON_SEND_KEYS = {
    "key", "title", "type", "tags", "source",
    "countries_with_data", "xaxis_title", "yaxis_title",
}


async def run_unified_analysis(
    query: str,
    query_focus: str,
    selected_tools: list[str],
    countries: list[str],
    comparison_manifest: list[dict[str, Any]],
    eda_chart_manifest: list[dict[str, Any]],
    eda_findings: dict[str, Any],
    data_coverage: dict[str, Any],
) -> dict[str, Any]:
    """Single LLM call returning chart keys (comparison + EDA) plus 3 hypotheses.

    Returns a dict with 3 keys:
      - comparison_chart_keys : list[str] (4 keys)
      - eda_chart_keys        : list[str] (0-4 keys)
      - hypotheses            : list[HypothesisInsight] (3 items — padded if LLM returns fewer)
    """
    client = get_genai_client()
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    # Only expose comparison charts that have data for ≥2 countries
    comp_available = [
        m for m in comparison_manifest if len(m.get("countries_with_data", [])) >= 2
    ]
    comp_safe = [
        {k: v for k, v in m.items() if k in _COMPARISON_SEND_KEYS}
        for m in comp_available
    ]

    eda_safe = [
        {
            "key": e["key"],
            "title": e["title"],
            "type":  e["type"],
            "description": e["description"],
            "metric": e.get("metric"),
            "applicable_countries": e.get("applicable_countries", []),
        }
        for e in eda_chart_manifest
    ]

    eda_summary = _build_eda_summary(eda_findings or {})

    prompt_parts = [
        f"USER QUERY: {query}",
        f"QUERY FOCUS: {query_focus or query}",
        f"COUNTRIES: {', '.join(countries)}",
        f"SELECTED TOOLS: {', '.join(selected_tools)}",
        "",
        "=== DATA COVERAGE (what columns came back per country) ===",
        json.dumps(data_coverage, indent=2),
        "",
        f"=== COMPARISON CHART MANIFEST ({len(comp_safe)} available) ===",
        json.dumps(comp_safe, indent=2),
        "",
        f"=== EDA CHART MANIFEST ({len(eda_safe)} available) ===",
        json.dumps(eda_safe, indent=2),
        "",
    ]
    if eda_summary:
        prompt_parts.extend(["=== EDA FINDINGS ===", eda_summary, ""])

    prompt_parts.append(
        "Return ONE JSON object with keys: comparison_chart_keys (4), "
        "eda_chart_keys (≤4), hypotheses (exactly 3)."
    )
    prompt = "\n".join(prompt_parts)

    comparison_keys: list[str] = []
    eda_keys: list[str] = []
    hypotheses: list[HypothesisInsight] = []

    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=UNIFIED_ANALYSIS_INSTRUCTION,
                response_mime_type="application/json",
                temperature=0.2,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        raw = json.loads(response.text)
        if not isinstance(raw, dict):
            raise ValueError("LLM did not return a JSON object")

        # Validate comparison keys: must exist in manifest, unique keys, unique metrics
        valid_comp_keys = {m["key"]: m for m in comp_available}
        seen_comp_metrics: set[str] = set()
        comparison_keys = []
        for k in (raw.get("comparison_chart_keys") or []):
            if not isinstance(k, str) or k not in valid_comp_keys:
                continue
            entry = valid_comp_keys[k]
            # Extract underlying metric from data_key or key prefix
            metric_id = entry.get("data_key") or k.rsplit("_", 1)[0]
            if metric_id in seen_comp_metrics:
                continue  # hard dedup: same metric already selected
            seen_comp_metrics.add(metric_id)
            comparison_keys.append(k)
            if len(comparison_keys) >= 4:
                break

        # Validate EDA keys: must exist in manifest, no metric appearing in comp section
        valid_eda_keys = {e["key"]: e for e in eda_chart_manifest}
        seen_eda_metrics: set[str] = set()
        eda_keys = []
        for k in (raw.get("eda_chart_keys") or []):
            if not isinstance(k, str) or k not in valid_eda_keys:
                continue
            entry = valid_eda_keys[k]
            metric = entry.get("metric") or k.split(":", 1)[-1]
            # Hard rule: no metric can appear in both EDA and comparison sections
            if metric in seen_comp_metrics:
                continue
            if metric in seen_eda_metrics:
                continue
            seen_eda_metrics.add(metric)
            eda_keys.append(k)
            if len(eda_keys) >= 4:
                break

        # Parse hypotheses
        raw_hyps = raw.get("hypotheses") or []
        if isinstance(raw_hyps, dict):
            raw_hyps = list(raw_hyps.values())
        hypotheses = [HypothesisInsight.model_validate(h) for h in raw_hyps[:3]]

    except Exception as exc:
        print(f"[UNIFIED_ANALYSIS] LLM failed: {exc}")

    # Pad hypotheses to exactly 3 with a placeholder
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

    return {
        "comparison_chart_keys": comparison_keys,
        "eda_chart_keys":        eda_keys,
        "hypotheses":            hypotheses[:3],
    }
