"""EDA Analyst: performs exploratory data analysis on collected country data.

Three classes of deterministic tool calls are made per run:
  1. run_growth_rate      — CAGR + peak / trough year for key economic series
  2. run_anomaly_detect   — Z-score outliers (|z| > 2) in every time series
  3. run_correlation_analysis — Pearson / Spearman across countries' latest values

The returned eda_findings dict is:
  • stored in ADK session state  →  consumed by ChartEvidenceADKAgent
  • surfaced to the browser via the "eda" SSE event
  • injected into the evidence-generator LLM prompt for grounded insights
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from analysis.correlation import run_correlation_analysis, run_growth_rate
from analysis.stats_tools import run_anomaly_detect

# ---------------------------------------------------------------------------
# Series extraction helpers
# ---------------------------------------------------------------------------

def _wb_series(dataset: dict, label: str) -> tuple[list[float], list[int]]:
    wb = pd.DataFrame(dataset.get("worldbank") or [])
    if wb.empty or "label" not in wb.columns:
        return [], []
    sub = wb[wb["label"] == label].dropna(subset=["value", "year"]).sort_values("year")
    if sub.empty:
        return [], []
    return sub["value"].astype(float).tolist(), sub["year"].astype(int).tolist()


def _emp_series(dataset: dict, col: str) -> tuple[list[float], list[int]]:
    emp = pd.DataFrame(dataset.get("employment") or [])
    if emp.empty or col not in emp.columns:
        return [], []
    sub = emp.dropna(subset=[col, "year"]).sort_values("year")
    if sub.empty:
        return [], []
    return sub[col].astype(float).tolist(), sub["year"].astype(int).tolist()


def _clim_series(dataset: dict, col: str) -> tuple[list[float], list[int]]:
    clim = pd.DataFrame(dataset.get("climate") or [])
    if clim.empty or col not in clim.columns:
        return [], []
    sub = clim.dropna(subset=[col, "year"]).sort_values("year")
    if sub.empty:
        return [], []
    return sub[col].astype(float).tolist(), sub["year"].astype(int).tolist()


def _disp_series(dataset: dict) -> tuple[list[float], list[int]]:
    disp = pd.DataFrame(dataset.get("displacement") or [])
    if disp.empty or "value" not in disp.columns or "year" not in disp.columns:
        return [], []
    yearly = disp.groupby("year")["value"].sum().sort_index()
    return yearly.values.astype(float).tolist(), yearly.index.astype(int).tolist()


# ---------------------------------------------------------------------------
# Metric extractor registry (used for cross-country matrix)
# ---------------------------------------------------------------------------

_METRIC_EXTRACTORS: dict[str, Any] = {
    "gdp_growth":          lambda d: _wb_series(d, "gdp_growth"),
    "inflation":           lambda d: _wb_series(d, "inflation"),
    "gdp_per_capita":      lambda d: _wb_series(d, "gdp_per_capita_usd"),
    "political_stability": lambda d: _wb_series(d, "political_stability"),
    "gini":                lambda d: _wb_series(d, "gini"),
    "unemployment":        lambda d: _emp_series(d, "unemployment_rate"),
    "youth_unemployment":  lambda d: _emp_series(d, "youth_unemployment_rate"),
    "temp_anomaly":        lambda d: _clim_series(d, "avg_temp_anomaly_c"),
    "precipitation":       lambda d: _clim_series(d, "annual_precipitation_mm"),
}


def _latest_value(vals: list[float]) -> float | None:
    for v in reversed(vals):
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            return float(v)
    return None


def _build_latest_matrix(countries_data: dict) -> pd.DataFrame:
    """Country × metric DataFrame of the latest available value per series."""
    rows = []
    for country, dataset in countries_data.items():
        row: dict[str, Any] = {"country": country}
        for metric, extractor in _METRIC_EXTRACTORS.items():
            vals, _ = extractor(dataset)
            row[metric] = _latest_value(vals)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("country")
    # Keep only columns where ≥ 3 countries (or ≥ all if fewer than 3) have data
    min_coverage = min(3, len(df))
    df = df.loc[:, df.notna().sum() >= min_coverage]
    return df


# ---------------------------------------------------------------------------
# Main EDA function
# ---------------------------------------------------------------------------

def run_eda(
    countries_data: dict[str, Any],
    selected_tools: list[str],
    query_focus: str,
) -> dict[str, Any]:
    """
    Perform exploratory data analysis on collected country data.

    Tool calls made deterministically:
      • run_growth_rate(vals, years)          → CAGR + peak/trough per series
      • run_anomaly_detect(vals, years)       → |z|>2 outlier years per series
      • run_correlation_analysis(data, col)   → Pearson/Spearman across metrics

    Returns:
      findings        – up to 4 narrative EDA findings (type, title, value, detail)
      growth_rates    – {country: {metric: growth_result}}
      anomalies       – {country: [{metric, years, z_scores}]}
      stats_summary   – {country: {metric: {mean, median, std, min, max, n}}}
      correlations    – top Pearson/Spearman correlation results
      charts_hint     – recommended EDA chart types for eda_charts.py
      latest_matrix   – {metric: {country: latest_value}}  (for heatmap)
    """
    findings: list[dict] = []
    growth_rates: dict[str, dict] = {}
    anomalies: dict[str, list] = {}
    stats_summary: dict[str, dict] = {}
    charts_hint: list[str] = []

    # Build the active metric list based on which tools were selected
    active_metrics: list[tuple[str, Any]] = [
        ("gdp_growth",    lambda d: _wb_series(d, "gdp_growth")),
        ("gdp_per_capita", lambda d: _wb_series(d, "gdp_per_capita_usd")),
        ("inflation",     lambda d: _wb_series(d, "inflation")),
    ]
    if "employment" in selected_tools:
        active_metrics += [
            ("unemployment",       lambda d: _emp_series(d, "unemployment_rate")),
            ("youth_unemployment", lambda d: _emp_series(d, "youth_unemployment_rate")),
        ]
    if "environment" in selected_tools:
        active_metrics += [
            ("temp_anomaly",  lambda d: _clim_series(d, "avg_temp_anomaly_c")),
            ("precipitation", lambda d: _clim_series(d, "annual_precipitation_mm")),
        ]
    if "unhcr" in selected_tools:
        active_metrics += [("displacement", lambda d: _disp_series(d))]

    # ── 1. Growth rate analysis (TOOL CALL: run_growth_rate) ──────────────────
    # We track CAGR for GDP per capita (a level series, always positive) — NOT
    # the gdp_growth rate, whose sign flips between years make CAGR undefined.
    per_capita_cagrs: dict[str, float] = {}

    for country, dataset in countries_data.items():
        growth_rates[country] = {}
        for metric_name, extractor in active_metrics:
            vals, years = extractor(dataset)
            if len(vals) >= 2:
                gr = run_growth_rate(vals, years)   # ← TOOL CALL #1
                growth_rates[country][metric_name] = gr
                if metric_name == "gdp_per_capita" and gr.get("cagr") is not None:
                    per_capita_cagrs[country] = gr["cagr"]

    # ── 2. Anomaly detection (TOOL CALL: run_anomaly_detect) ──────────────────
    anomaly_checks = [
        ("gdp_growth",   lambda d: _wb_series(d, "gdp_growth")),
        ("inflation",    lambda d: _wb_series(d, "inflation")),
        ("unemployment", lambda d: _emp_series(d, "unemployment_rate")),
        ("temp_anomaly", lambda d: _clim_series(d, "avg_temp_anomaly_c")),
    ]

    for country, dataset in countries_data.items():
        anomalies[country] = []
        for metric_name, extractor in anomaly_checks:
            vals, years = extractor(dataset)
            if len(vals) >= 5:
                result = run_anomaly_detect(vals, years)  # ← TOOL CALL #2
                if result.get("anomaly_years"):
                    anomalies[country].append({
                        "metric":   metric_name,
                        "years":    result["anomaly_years"],
                        "z_scores": result.get("z_scores", []),
                    })

    # ── 3. Statistical summary ────────────────────────────────────────────────
    for country, dataset in countries_data.items():
        stats_summary[country] = {}
        for metric_name, extractor in active_metrics:
            vals, _ = extractor(dataset)
            if vals:
                arr = np.asarray(vals, dtype=float)
                arr = arr[~np.isnan(arr)]
                if len(arr) > 0:
                    stats_summary[country][metric_name] = {
                        "mean":   float(np.mean(arr)),
                        "median": float(np.median(arr)),
                        "std":    float(np.std(arr)),
                        "min":    float(np.min(arr)),
                        "max":    float(np.max(arr)),
                        "n":      int(len(arr)),
                    }

    # ── 4. Cross-country correlation (TOOL CALL: run_correlation_analysis) ────
    correlation_results: list[dict] = []
    latest_matrix = _build_latest_matrix(countries_data)

    if (
        not latest_matrix.empty
        and len(latest_matrix.columns) >= 2
        and len(latest_matrix) >= 3
    ):
        complete = latest_matrix.dropna()
        if len(complete) >= 3 and len(complete.columns) >= 2:
            corr_input = {col: complete[col].tolist() for col in complete.columns}
            target_col = (
                "gdp_growth" if "gdp_growth" in complete.columns
                else complete.columns[0]
            )
            correlation_results = run_correlation_analysis(corr_input, target_col)  # ← TOOL CALL #3

    # ── 5. Surface narrative findings ─────────────────────────────────────────

    # (a) GDP per capita growth leader / laggard (CAGR on level series)
    if per_capita_cagrs:
        valid = {c: v for c, v in per_capita_cagrs.items() if v is not None}
        if valid:
            leader  = max(valid, key=lambda c: valid[c])
            laggard = min(valid, key=lambda c: valid[c])
            lv  = valid[leader]
            lgv = valid[laggard]
            findings.append({
                "type":   "growth",
                "title":  f"{leader} leads GDP-per-capita trajectory",
                "value":  f"CAGR {lv * 100:+.1f}%",
                "detail": (
                    f"{leader} shows the highest compound annual GDP-per-capita "
                    f"growth ({lv * 100:+.1f}%) across the analysis window."
                    + (
                        f" {laggard} records the weakest at {lgv * 100:+.1f}% CAGR."
                        if leader != laggard else ""
                    )
                ),
            })
            charts_hint.append("growth_rate_bar")

    # (b) First anomaly found across any country
    any_anomalies = {c: a for c, a in anomalies.items() if a}
    if any_anomalies:
        country, anom_list = next(iter(any_anomalies.items()))
        a = anom_list[0]
        year_str = ", ".join(str(y) for y in a["years"])
        findings.append({
            "type":   "anomaly",
            "title":  f"Statistical outlier: {country} {a['metric'].replace('_', ' ')}",
            "value":  f"Outlier year(s): {year_str}",
            "detail": (
                f"{country} shows a statistically significant deviation (|z| > 2) "
                f"in {a['metric'].replace('_', ' ')} during {year_str}. "
                "This likely corresponds to a crisis event or structural break."
            ),
        })
        charts_hint.append("anomaly_timeline")

    # (c) Strongest correlation finding
    if correlation_results:
        top = correlation_results[0]
        if abs(top["pearson_r"]) >= 0.4:
            direction = "positive" if top["pearson_r"] > 0 else "negative"
            n_countries = len(countries_data)
            findings.append({
                "type":   "correlation",
                "title":  f"{direction.capitalize()} link: {top['indicator'].replace('_', ' ')} ↔ GDP growth",
                "value":  f"r = {top['pearson_r']:.2f}  (p = {top['pearson_p']:.3f})",
                "detail": (
                    f"Across {n_countries} countries, {top['indicator'].replace('_', ' ')} "
                    f"has a Pearson r of {top['pearson_r']:.2f} with GDP growth, "
                    f"indicating a {direction} cross-country relationship."
                ),
            })
            charts_hint.append("correlation_heatmap")

    # (d) Highest volatility country
    for metric_name in ["gdp_growth", "unemployment", "inflation"]:
        stds = {
            c: stats_summary[c][metric_name]["std"]
            for c in stats_summary
            if stats_summary[c].get(metric_name, {}).get("std") is not None
        }
        if len(stds) >= 2:
            most_volatile = max(stds, key=lambda c: stds[c])
            findings.append({
                "type":   "volatility",
                "title":  f"Most volatile {metric_name.replace('_', ' ')}: {most_volatile}",
                "value":  f"σ = {stds[most_volatile]:.2f}",
                "detail": (
                    f"{most_volatile} shows the highest year-to-year variability "
                    f"in {metric_name.replace('_', ' ')} (σ = {stds[most_volatile]:.2f}), "
                    "suggesting less predictable economic conditions vs. peers."
                ),
            })
            charts_hint.append("distribution_box")
            break

    if not charts_hint:
        charts_hint = ["growth_rate_bar"]

    return {
        "findings":       findings[:4],
        "growth_rates":   growth_rates,
        "anomalies":      anomalies,
        "stats_summary":  stats_summary,
        "correlations":   correlation_results[:6],
        "charts_hint":    list(dict.fromkeys(charts_hint)),   # dedup, preserve order
        "latest_matrix":  latest_matrix.to_dict() if not latest_matrix.empty else {},
    }
