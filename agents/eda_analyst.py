"""EDA Analyst: query-aware exploratory data analysis on collected country data.

Metrics are chosen entirely from selected_tools so the analysis always reflects
the actual query topic:
  - safety/conflict query  → conflict events, fatalities, political stability
  - economic query         → GDP, inflation, income growth
  - labour query           → unemployment, youth unemployment
  - environment query      → temperature anomaly, precipitation
  - displacement query     → UNHCR outflow trends

Statistical methods used (all distinct from the comparison line/bar charts below):
  1. run_growth_rate       → CAGR + peak/trough year per series
  2. run_anomaly_detect    → Z-score |z|>2 outlier detection per series
  3. run_correlation_analysis → Pearson/Spearman cross-country correlation matrix
  4. Statistical spread    → mean ± σ per country

Charts hints always prefer distribution_box and correlation_heatmap first —
purely statistical visuals that look nothing like the time-series comparison charts.
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


def _conflict_events_series(dataset: dict) -> tuple[list[float], list[int]]:
    """Annual count of conflict events (ACLED)."""
    conf = pd.DataFrame(dataset.get("conflict_events") or [])
    if conf.empty:
        return [], []
    if "year" not in conf.columns:
        if "event_date" in conf.columns:
            conf = conf.copy()
            conf["year"] = pd.to_datetime(conf["event_date"], errors="coerce").dt.year
        else:
            return [], []
    conf = conf.dropna(subset=["year"])
    yearly = conf.groupby("year").size().sort_index()
    return yearly.values.astype(float).tolist(), yearly.index.astype(int).tolist()


def _fatalities_series(dataset: dict) -> tuple[list[float], list[int]]:
    """Annual fatality totals (ACLED)."""
    conf = pd.DataFrame(dataset.get("conflict_events") or [])
    if conf.empty or "fatalities" not in conf.columns:
        return [], []
    if "year" not in conf.columns:
        if "event_date" in conf.columns:
            conf = conf.copy()
            conf["year"] = pd.to_datetime(conf["event_date"], errors="coerce").dt.year
        else:
            return [], []
    conf = conf.dropna(subset=["year", "fatalities"])
    yearly = conf.groupby("year")["fatalities"].sum().sort_index()
    return yearly.values.astype(float).tolist(), yearly.index.astype(int).tolist()


# ---------------------------------------------------------------------------
# Tool-driven metric registry
# ---------------------------------------------------------------------------

# (extractor_fn, human_label)
_ALL_METRICS: dict[str, tuple[Any, str]] = {
    "conflict_events":     (_conflict_events_series,                               "Conflict events"),
    "fatalities":          (_fatalities_series,                                    "Fatalities"),
    "political_stability": (lambda d: _wb_series(d, "political_stability"),        "Political stability index"),
    "displacement":        (_disp_series,                                          "Displacement outflow"),
    "unemployment":        (lambda d: _emp_series(d, "unemployment_rate"),         "Unemployment rate (%)"),
    "youth_unemployment":  (lambda d: _emp_series(d, "youth_unemployment_rate"),   "Youth unemployment (%)"),
    "temp_anomaly":        (lambda d: _clim_series(d, "avg_temp_anomaly_c"),       "Temp anomaly (°C)"),
    "precipitation":       (lambda d: _clim_series(d, "annual_precipitation_mm"), "Precipitation (mm)"),
    "gdp_growth":              (lambda d: _wb_series(d, "gdp_growth"),                  "GDP growth (%)"),
    "gdp_per_capita":          (lambda d: _wb_series(d, "gdp_per_capita_usd"),          "GDP per capita (USD)"),
    "inflation":               (lambda d: _wb_series(d, "inflation"),                   "Inflation (%)"),
    "gini":                    (lambda d: _wb_series(d, "gini"),                        "Gini index"),
    "health_expenditure_gdp":  (lambda d: _wb_series(d, "health_expenditure_gdp"),      "Health expenditure (% GDP)"),
    "physicians_per_1000":     (lambda d: _wb_series(d, "physicians_per_1000"),         "Physicians per 1,000 people"),
    "education_spend_gdp":     (lambda d: _wb_series(d, "education_spend_gdp"),         "Education spending (% GDP)"),
    "poverty_headcount":       (lambda d: _wb_series(d, "poverty_headcount"),           "Poverty headcount (%)"),
}

# Which metrics to activate per tool, in priority order within that tool
_TOOL_METRICS: dict[str, list[str]] = {
    "acled":       ["conflict_events", "fatalities"],
    "unhcr":       ["displacement"],
    "worldbank":   ["political_stability", "gdp_growth", "gdp_per_capita", "inflation", "gini",
                   "health_expenditure_gdp", "physicians_per_1000", "education_spend_gdp", "poverty_headcount"],
    "employment":  ["unemployment", "youth_unemployment"],
    "environment": ["temp_anomaly", "precipitation"],
    "news":        [],
}

# Primary metric to use as correlation anchor per tool
_TOOL_ANCHOR: dict[str, str] = {
    "acled":       "conflict_events",
    "unhcr":       "displacement",
    "environment": "temp_anomaly",
    "employment":  "unemployment",
    "worldbank":   "political_stability",
}

# Human-readable migration implication per metric (for volatility finding)
_METRIC_VOLATILITY_IMPLICATIONS: dict[str, str] = {
    "conflict_events":     "unpredictable security — conflict levels can spike suddenly",
    "fatalities":          "highly unstable conflict intensity — fatalities vary dramatically year to year",
    "displacement":        "unstable migration pressure — outflows fluctuate significantly",
    "political_stability": "unstable governance — political conditions shift frequently",
    "unemployment":        "unstable employment — boom-and-bust hiring cycles",
    "youth_unemployment":  "volatile youth job market — entry-level prospects are hard to predict",
    "inflation":           "erratic cost of living — purchasing power can shift sharply",
    "gdp_growth":          "unpredictable income growth — economic conditions are inconsistent",
    "temp_anomaly":            "high climate variability — conditions are becoming less predictable",
    "health_expenditure_gdp":  "inconsistent healthcare investment — access to public health services may be unreliable",
    "physicians_per_1000":     "uneven healthcare workforce — medical access varies year to year",
    "education_spend_gdp":     "volatile education funding — quality of schooling may fluctuate significantly",
    "poverty_headcount":       "unstable poverty levels — economic conditions shift for the most vulnerable",
}


def _build_active_metrics(selected_tools: list[str]) -> list[tuple[str, Any]]:
    """Return (metric_name, extractor) pairs ordered by tool priority."""
    seen: set[str] = set()
    result: list[tuple[str, Any]] = []
    for tool in selected_tools:
        for metric in _TOOL_METRICS.get(tool, []):
            if metric not in seen and metric in _ALL_METRICS:
                seen.add(metric)
                extractor, _ = _ALL_METRICS[metric]
                result.append((metric, extractor))
    return result


def _primary_anchor(selected_tools: list[str], available: set[str]) -> str:
    """Pick the most query-relevant correlation anchor."""
    for tool in selected_tools:
        candidate = _TOOL_ANCHOR.get(tool, "")
        if candidate and candidate in available:
            return candidate
    for fallback in ["political_stability", "gdp_growth", "unemployment"]:
        if fallback in available:
            return fallback
    return next(iter(available)) if available else "gdp_growth"


def _latest_value(vals: list[float]) -> float | None:
    for v in reversed(vals):
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            return float(v)
    return None


def _build_latest_matrix(
    countries_data: dict,
    active_metrics: list[tuple[str, Any]],
) -> pd.DataFrame:
    """Country × metric DataFrame of the latest available value per series."""
    rows = []
    for country, dataset in countries_data.items():
        row: dict[str, Any] = {"country": country}
        for metric, extractor in active_metrics:
            vals, _ = extractor(dataset)
            row[metric] = _latest_value(vals)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("country")
    min_coverage = min(3, len(df))
    df = df.loc[:, df.notna().sum() >= min_coverage]
    return df


# ---------------------------------------------------------------------------
# Narrative finding generators
# ---------------------------------------------------------------------------

def _finding_conflict_spread(countries_data: dict) -> dict | None:
    """Cross-sectional: which country has most vs least conflict events."""
    totals: dict[str, float] = {}
    for country, dataset in countries_data.items():
        vals, _ = _conflict_events_series(dataset)
        if vals:
            totals[country] = float(np.sum(vals))
    if len(totals) < 2:
        return None
    worst  = max(totals, key=lambda c: totals[c])
    safest = min(totals, key=lambda c: totals[c])
    ratio  = totals[worst] / totals[safest] if totals[safest] > 0 else None
    ratio_note = f" — {ratio:.0f}× more than {safest}" if ratio and ratio >= 2 else f" compared to {int(totals[safest]):,} in {safest}"
    return {
        "type":   "volatility",
        "title":  f"{worst} recorded the most conflict events",
        "value":  f"{int(totals[worst]):,} total events",
        "detail": (
            f"Over the analysis period, {worst} recorded {int(totals[worst]):,} conflict events"
            f"{ratio_note}. "
            f"For migrants prioritising safety, this gap is a critical signal: "
            f"lower event counts indicate reduced exposure to violence and civil disruption."
        ),
    }


def _finding_fatality_spread(countries_data: dict) -> dict | None:
    """Cross-sectional: total fatalities comparison."""
    totals: dict[str, float] = {}
    for country, dataset in countries_data.items():
        vals, _ = _fatalities_series(dataset)
        if vals:
            totals[country] = float(np.sum(vals))
    if len(totals) < 2:
        return None
    worst  = max(totals, key=lambda c: totals[c])
    safest = min(totals, key=lambda c: totals[c])
    return {
        "type":   "volatility",
        "title":  f"{worst} had the highest conflict fatalities",
        "value":  f"{int(totals[worst]):,} fatalities total",
        "detail": (
            f"{worst} recorded {int(totals[worst]):,} conflict-related fatalities "
            f"over the analysis window — far higher than {safest} ({int(totals[safest]):,}). "
            f"This is the most direct measure of lethal violence and is a primary "
            f"safety factor for migrants considering relocation."
        ),
    }


def _finding_stability_ranking(stats_summary: dict) -> dict | None:
    """World Bank political stability: best vs worst country."""
    means = {
        c: stats_summary[c]["political_stability"]["mean"]
        for c in stats_summary
        if stats_summary[c].get("political_stability", {}).get("mean") is not None
    }
    if len(means) < 2:
        return None
    best  = max(means, key=lambda c: means[c])
    worst = min(means, key=lambda c: means[c])
    return {
        "type":   "growth",
        "title":  f"{best} ranks highest on political stability",
        "value":  f"Index {means[best]:.2f}  (vs {means[worst]:.2f} for {worst})",
        "detail": (
            f"The World Bank political stability index places {best} at {means[best]:.2f} "
            f"on average — the strongest score in this group. "
            f"{worst} scores {means[worst]:.2f}, indicating higher perceived risk of "
            f"instability and political violence. "
            f"This index directly reflects a country's safety and predictability for migrants."
        ),
    }


def _finding_displacement_trend(growth_rates: dict) -> dict | None:
    """Country with fastest-growing displacement outflow."""
    cagrs = {
        c: growth_rates[c]["displacement"]["cagr"]
        for c in growth_rates
        if growth_rates[c].get("displacement", {}).get("cagr") is not None
    }
    if len(cagrs) < 2:
        return None
    highest = max(cagrs, key=lambda c: cagrs[c])
    lowest  = min(cagrs, key=lambda c: cagrs[c])
    hv, lv  = cagrs[highest], cagrs[lowest]
    trend_note = (
        f"{lowest}'s outflow is shrinking ({lv * 100:+.1f}% CAGR), "
        f"suggesting stabilising conditions there."
        if lv < 0 else
        f"{lowest} shows the most contained outflow growth ({lv * 100:+.1f}% CAGR)."
    )
    return {
        "type":   "growth",
        "title":  f"{highest} has fastest-rising displacement outflow",
        "value":  f"CAGR {hv * 100:+.1f}% per year",
        "detail": (
            f"{highest}'s displacement outflow is growing {hv * 100:+.1f}% per year — "
            f"the fastest among compared countries. Rising displacement is a strong "
            f"push-factor signal indicating worsening conditions. "
            f"{trend_note}"
        ),
    }


def _finding_income_growth(per_capita_cagrs: dict) -> dict | None:
    """Economic fallback: GDP per capita CAGR comparison."""
    valid = {c: v for c, v in per_capita_cagrs.items() if v is not None}
    if len(valid) < 2:
        return None
    leader  = max(valid, key=lambda c: valid[c])
    laggard = min(valid, key=lambda c: valid[c])
    lv, lgv = valid[leader], valid[laggard]
    return {
        "type":   "growth",
        "title":  f"{leader} leads income growth",
        "value":  f"CAGR {lv * 100:+.1f}% per year",
        "detail": (
            f"{leader}'s GDP per capita has grown {lv * 100:+.1f}% annually — "
            f"the fastest among compared countries. "
            f"For migrants, this signals rising living standards and stronger wage prospects. "
            f"{laggard} grew only {lgv * 100:+.1f}% per year — "
            f"meaning purchasing power there is rising much more slowly."
            if leader != laggard else
            f"{leader}'s GDP per capita has grown {lv * 100:+.1f}% annually."
        ),
    }


def _finding_anomaly(anomalies: dict, selected_tools: list[str]) -> dict | None:
    """Surface the anomaly in the most query-relevant metric."""
    priority: list[str] = []
    for tool in selected_tools:
        priority.extend(_TOOL_METRICS.get(tool, []))
    # Generic fallbacks
    priority += ["gdp_growth", "inflation", "unemployment", "temp_anomaly"]

    for target_metric in priority:
        for country, anom_list in anomalies.items():
            for a in anom_list:
                if a["metric"] == target_metric:
                    year_str = ", ".join(str(y) for y in a["years"])
                    _, label = _ALL_METRICS.get(target_metric, (None, target_metric.replace("_", " ")))
                    return {
                        "type":   "anomaly",
                        "title":  f"Unusual spike: {country} — {label}",
                        "value":  f"Outlier year(s): {year_str}",
                        "detail": (
                            f"In {year_str}, {country}'s {label.lower()} moved more than "
                            f"2 standard deviations from its own historical average. "
                            f"Such spikes often signal a crisis, escalation event, or "
                            f"sudden policy change — conditions that directly affect "
                            f"safety, employment, and quality of life for migrants."
                        ),
                    }
    return None


def _finding_correlation(
    correlation_results: list[dict],
    anchor_metric: str,
    n_countries: int,
) -> dict | None:
    if not correlation_results:
        return None
    top = correlation_results[0]
    if abs(top["pearson_r"]) < 0.4:
        return None
    direction = "positive" if top["pearson_r"] > 0 else "negative"
    indicator_label = top["indicator"].replace("_", " ")
    _, anchor_label = _ALL_METRICS.get(anchor_metric, (None, anchor_metric.replace("_", " ")))
    if direction == "positive":
        implication = (
            f"Countries where {indicator_label} is higher also tend to have "
            f"higher {anchor_label.lower()} — both factors track together in this group."
        )
    else:
        implication = (
            f"Countries where {indicator_label} is higher tend to have lower "
            f"{anchor_label.lower()} — a trade-off migrants should weigh when choosing a destination."
        )
    return {
        "type":   "correlation",
        "title":  f"{indicator_label.title()} correlates with {anchor_label.lower()}",
        "value":  f"r = {top['pearson_r']:.2f}  (p = {top['pearson_p']:.3f})",
        "detail": (
            f"Across {n_countries} countries, {indicator_label} and {anchor_label.lower()} "
            f"move in a {direction} direction (Pearson r = {top['pearson_r']:.2f}). "
            f"{implication}"
        ),
    }


def _finding_volatility(stats_summary: dict, selected_tools: list[str]) -> dict | None:
    """Most volatile metric in query-priority order."""
    priority: list[str] = []
    for tool in selected_tools:
        priority.extend(_TOOL_METRICS.get(tool, []))

    for metric_name in priority:
        stds = {
            c: stats_summary[c][metric_name]["std"]
            for c in stats_summary
            if stats_summary[c].get(metric_name, {}).get("std") is not None
        }
        if len(stds) < 2:
            continue
        most_volatile  = max(stds, key=lambda c: stds[c])
        least_volatile = min(stds, key=lambda c: stds[c])
        _, label = _ALL_METRICS.get(metric_name, (None, metric_name.replace("_", " ")))
        implication = _METRIC_VOLATILITY_IMPLICATIONS.get(metric_name, "less predictable conditions")
        stable_note = (
            f" {least_volatile} shows the most consistent {label.lower()} "
            f"(σ = {stds[least_volatile]:.2f}) — the more predictable choice."
            if most_volatile != least_volatile else ""
        )
        return {
            "type":   "volatility",
            "title":  f"{most_volatile} has the most variable {label.lower()}",
            "value":  f"σ = {stds[most_volatile]:.2f}",
            "detail": (
                f"{most_volatile}'s {label.lower()} fluctuates more than any other "
                f"country in this comparison (standard deviation: {stds[most_volatile]:.2f}). "
                f"For migrants, this signals {implication}.{stable_note}"
            ),
        }
    return None


# ---------------------------------------------------------------------------
# Main EDA function
# ---------------------------------------------------------------------------

def run_eda(
    countries_data: dict[str, Any],
    selected_tools: list[str],
) -> dict[str, Any]:
    """
    Query-aware EDA: metric set is driven entirely by selected_tools.

    For a safety query (acled selected): conflict events, fatalities, political stability.
    For an economic query (worldbank): GDP, inflation, income.
    For a labour query (employment): unemployment, youth unemployment.
    For environment: temperature anomaly, precipitation.
    For displacement (unhcr): outflow trends.

    Chart hints always prefer distribution_box and correlation_heatmap first —
    purely statistical visuals distinct from the time-series comparison charts.
    """
    findings: list[dict] = []
    growth_rates: dict[str, dict] = {}
    anomalies: dict[str, list] = {}
    stats_summary: dict[str, dict] = {}
    charts_hint: list[str] = []

    # ── Build tool-driven active metric list ──────────────────────────────────
    active_metrics = _build_active_metrics(selected_tools)
    if not active_metrics:
        # Absolute fallback
        active_metrics = [
            ("gdp_growth",    lambda d: _wb_series(d, "gdp_growth")),
            ("gdp_per_capita", lambda d: _wb_series(d, "gdp_per_capita_usd")),
            ("inflation",     lambda d: _wb_series(d, "inflation")),
        ]

    # ── 1. Growth rates (CAGR) ────────────────────────────────────────────────
    per_capita_cagrs: dict[str, float] = {}
    for country, dataset in countries_data.items():
        growth_rates[country] = {}
        for metric_name, extractor in active_metrics:
            vals, years = extractor(dataset)
            if len(vals) >= 2:
                gr = run_growth_rate(vals, years)
                growth_rates[country][metric_name] = gr
                if metric_name == "gdp_per_capita" and gr.get("cagr") is not None:
                    per_capita_cagrs[country] = gr["cagr"]

    # ── 2. Anomaly detection (z-score |z|>2) ─────────────────────────────────
    for country, dataset in countries_data.items():
        anomalies[country] = []
        for metric_name, extractor in active_metrics:
            vals, years = extractor(dataset)
            if len(vals) >= 5:
                result = run_anomaly_detect(vals, years)
                if result.get("anomaly_years"):
                    anomalies[country].append({
                        "metric":   metric_name,
                        "years":    result["anomaly_years"],
                        "z_scores": result.get("z_scores", []),
                    })

    # ── 3. Statistical summary (mean / median / std / min / max) ─────────────
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

    # ── 4. Cross-country correlation ─────────────────────────────────────────
    correlation_results: list[dict] = []
    latest_matrix = _build_latest_matrix(countries_data, active_metrics)
    anchor_metric = _primary_anchor(selected_tools, set(latest_matrix.columns.tolist()))

    if (
        not latest_matrix.empty
        and len(latest_matrix.columns) >= 2
        and len(latest_matrix) >= 3
    ):
        complete = latest_matrix.dropna()
        if len(complete) >= 3 and len(complete.columns) >= 2:
            corr_input = {col: complete[col].tolist() for col in complete.columns}
            target_col = anchor_metric if anchor_metric in complete.columns else complete.columns[0]
            correlation_results = run_correlation_analysis(corr_input, target_col)

    # ── 5. Narrative findings — tool-priority order ───────────────────────────

    # ACLED: conflict event count spread (cross-sectional bar)
    if "acled" in selected_tools:
        f = _finding_conflict_spread(countries_data)
        if f:
            findings.append(f)
            charts_hint.append("distribution_box")

    # ACLED: fatality spread (if conflict finding didn't fire or we still have room)
    if "acled" in selected_tools and len(findings) < 2:
        f = _finding_fatality_spread(countries_data)
        if f:
            findings.append(f)
            charts_hint.append("distribution_box")

    # WORLDBANK: political stability ranking
    if "worldbank" in selected_tools and len(findings) < 3:
        f = _finding_stability_ranking(stats_summary)
        if f:
            findings.append(f)
            charts_hint.append("distribution_box")

    # UNHCR: displacement trend (CAGR)
    if "unhcr" in selected_tools and len(findings) < 3:
        f = _finding_displacement_trend(growth_rates)
        if f:
            findings.append(f)
            charts_hint.append("growth_rate_bar")

    # Correlation (heatmap — always visually distinct)
    if len(findings) < 4:
        f = _finding_correlation(correlation_results, anchor_metric, len(countries_data))
        if f:
            findings.append(f)
            charts_hint.append("correlation_heatmap")

    # Anomaly detection (z-score spike on the most relevant metric)
    if len(findings) < 4:
        f = _finding_anomaly(anomalies, selected_tools)
        if f:
            findings.append(f)
            charts_hint.append("anomaly_timeline")

    # Volatility spread (mean ± σ bar — purely statistical)
    if len(findings) < 4:
        f = _finding_volatility(stats_summary, selected_tools)
        if f:
            findings.append(f)
            charts_hint.append("distribution_box")

    # Economic income growth (only if not already covered and worldbank is selected)
    if len(findings) < 2 and "worldbank" in selected_tools:
        f = _finding_income_growth(per_capita_cagrs)
        if f:
            findings.append(f)
            charts_hint.append("growth_rate_bar")

    # Guarantee at least one statistical chart (not time-series)
    if "distribution_box" not in charts_hint and "correlation_heatmap" not in charts_hint:
        charts_hint = ["distribution_box"] + charts_hint

    return {
        "findings":       findings[:4],
        "growth_rates":   growth_rates,
        "anomalies":      anomalies,
        "stats_summary":  stats_summary,
        "correlations":   correlation_results[:6],
        "charts_hint":    list(dict.fromkeys(charts_hint))[:2],  # max 2, deduped
        "latest_matrix":  latest_matrix.to_dict() if not latest_matrix.empty else {},
    }
