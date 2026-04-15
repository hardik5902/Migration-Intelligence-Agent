"""EDA Analyst: query-aware exploratory data analysis on collected country data.

Metrics are chosen entirely from selected_tools so the analysis always reflects
the actual query topic:
  - safety/conflict query  → conflict events, fatalities, political stability
  - economic query         → GDP, inflation, income growth
  - labour query           → unemployment, youth unemployment
  - environment query      → temperature anomaly, precipitation


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


def _teleport_series(dataset: dict) -> tuple[list[float], list[int]]:
    """Composite quality-of-life score from Teleport / World Bank proxy (latest value repeated)."""
    cs = pd.DataFrame(dataset.get("city_scores") or [])
    if cs.empty or "score_out_of_10" not in cs.columns:
        return [], []
    vals = cs["score_out_of_10"].dropna()
    if vals.empty:
        return [], []
    # city_scores may not have a year column — use a single aggregate point
    if "year" in cs.columns:
        sub = cs.dropna(subset=["score_out_of_10", "year"])
        if not sub.empty:
            grouped = sub.groupby("year")["score_out_of_10"].mean().sort_index()
            return grouped.values.astype(float).tolist(), grouped.index.astype(int).tolist()
    # No year column — return a single mean value tagged to 2023
    return [float(vals.mean())], [2023]


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
    # ── Conflict / safety ──────────────────────────────────────────────────────
    "conflict_events":         (_conflict_events_series,                                    "Conflict events"),
    "fatalities":              (_fatalities_series,                                         "Fatalities"),
    # ── Governance (World Bank WGI) ────────────────────────────────────────────
    "political_stability":     (lambda d: _wb_series(d, "political_stability"),             "Political stability index"),
    "control_of_corruption":   (lambda d: _wb_series(d, "control_of_corruption"),           "Control of corruption index"),
    "rule_of_law":             (lambda d: _wb_series(d, "rule_of_law"),                     "Rule of law index"),
    "homicide_rate":           (lambda d: _wb_series(d, "homicide_rate"),                   "Homicide rate (per 100k)"),
    # ── Labour ────────────────────────────────────────────────────────────────
    "unemployment":            (lambda d: _emp_series(d, "unemployment_rate"),              "Unemployment rate (%)"),
    "youth_unemployment":      (lambda d: _emp_series(d, "youth_unemployment_rate"),        "Youth unemployment (%)"),
    "female_labor_participation": (lambda d: _wb_series(d, "female_labor_participation"),   "Female labour participation (%)"),
    # ── Economy ───────────────────────────────────────────────────────────────
    "gdp_growth":              (lambda d: _wb_series(d, "gdp_growth"),                      "GDP growth (%)"),
    "gdp_per_capita":          (lambda d: _wb_series(d, "gdp_per_capita_usd"),              "GDP per capita (USD)"),
    "gni_per_capita_ppp":      (lambda d: _wb_series(d, "gni_per_capita_ppp"),              "GNI per capita PPP (int. $)"),
    "inflation":               (lambda d: _wb_series(d, "inflation"),                       "Inflation (%)"),
    "gini":                    (lambda d: _wb_series(d, "gini"),                            "Gini index"),
    "poverty_headcount":       (lambda d: _wb_series(d, "poverty_headcount"),               "Poverty headcount (%)"),
    # ── Health ────────────────────────────────────────────────────────────────
    "health_expenditure_gdp":  (lambda d: _wb_series(d, "health_expenditure_gdp"),          "Health expenditure (% GDP)"),
    "physicians_per_1000":     (lambda d: _wb_series(d, "physicians_per_1000"),             "Physicians per 1,000 people"),
    "life_expectancy":         (lambda d: _wb_series(d, "life_expectancy"),                 "Life expectancy (years)"),
    "infant_mortality":        (lambda d: _wb_series(d, "infant_mortality"),                "Infant mortality (per 1,000 births)"),
    # ── Education ─────────────────────────────────────────────────────────────
    "education_spend_gdp":     (lambda d: _wb_series(d, "education_spend_gdp"),             "Education spending (% GDP)"),
    # ── Infrastructure / access ───────────────────────────────────────────────
    "sanitation_access":       (lambda d: _wb_series(d, "sanitation_access"),               "Sanitation access (% pop.)"),
    "clean_water_access":      (lambda d: _wb_series(d, "clean_water_access"),              "Clean water access (% pop.)"),
    "electricity_access":      (lambda d: _wb_series(d, "electricity_access"),              "Electricity access (% pop.)"),
    "internet_users_pct":      (lambda d: _wb_series(d, "internet_users_pct"),              "Internet users (% pop.)"),
    "urban_population_pct":    (lambda d: _wb_series(d, "urban_population_pct"),            "Urban population (% total)"),
    # ── Gender ────────────────────────────────────────────────────────────────
    "women_in_parliament":     (lambda d: _wb_series(d, "women_in_parliament"),             "Women in parliament (%)"),
    "adolescent_fertility_rate": (lambda d: _wb_series(d, "adolescent_fertility_rate"),     "Adolescent fertility rate"),
    # ── Environment ───────────────────────────────────────────────────────────
    "co2_per_capita":          (lambda d: _wb_series(d, "co2_per_capita"),                  "CO₂ emissions per capita (t)"),
    "temp_anomaly":            (lambda d: _clim_series(d, "avg_temp_anomaly_c"),            "Temp anomaly (°C)"),
    "precipitation":           (lambda d: _clim_series(d, "annual_precipitation_mm"),      "Precipitation (mm)"),
    # ── Quality of life ───────────────────────────────────────────────────────
    "teleport_score":          (_teleport_series,                                           "Quality-of-life score (0–10)"),
}

# Which metrics to activate per tool, in priority order within that tool
_TOOL_METRICS: dict[str, list[str]] = {
    "acled":       ["conflict_events", "fatalities"],
    "worldbank":   [
        # Governance
        "political_stability", "control_of_corruption", "rule_of_law", "homicide_rate",
        # Economy
        "gdp_growth", "gdp_per_capita", "gni_per_capita_ppp", "inflation", "gini", "poverty_headcount",
        # Health
        "health_expenditure_gdp", "physicians_per_1000", "life_expectancy", "infant_mortality",
        # Education
        "education_spend_gdp",
        # Infrastructure / access
        "sanitation_access", "clean_water_access", "electricity_access", "internet_users_pct", "urban_population_pct",
        # Gender
        "female_labor_participation", "women_in_parliament", "adolescent_fertility_rate",
        # Environment
        "co2_per_capita",
    ],
    "employment":  ["unemployment", "youth_unemployment", "female_labor_participation"],
    "environment": ["temp_anomaly", "precipitation", "co2_per_capita"],
    "teleport":    ["teleport_score"],
    "news":        [],
}

# Primary metric to use as correlation anchor per tool
_TOOL_ANCHOR: dict[str, str] = {
    "acled":       "conflict_events",
    "environment": "temp_anomaly",
    "employment":  "unemployment",
    "worldbank":   "political_stability",
    "teleport":    "teleport_score",
}

# Human-readable migration implication per metric (for volatility finding)
_METRIC_VOLATILITY_IMPLICATIONS: dict[str, str] = {
    # Conflict / safety
    "conflict_events":           "unpredictable security — conflict levels can spike suddenly",
    "fatalities":                "highly unstable conflict intensity — fatalities vary dramatically year to year",
    # Governance
    "political_stability":       "unstable governance — political conditions shift frequently",
    "control_of_corruption":     "inconsistent anti-corruption enforcement — business and legal environments are hard to predict",
    "rule_of_law":               "volatile rule-of-law scores — legal protection for migrants and residents is uneven",
    "homicide_rate":             "fluctuating lethal violence — personal safety conditions are unpredictable",
    # Labour
    "unemployment":              "unstable employment — boom-and-bust hiring cycles",
    "youth_unemployment":        "volatile youth job market — entry-level prospects are hard to predict",
    "female_labor_participation":"uneven gender integration — women's economic opportunity shifts year to year",
    # Economy
    "inflation":                 "erratic cost of living — purchasing power can shift sharply",
    "gdp_growth":                "unpredictable income growth — economic conditions are inconsistent",
    "gni_per_capita_ppp":        "volatile real incomes — purchasing-power-adjusted living standards fluctuate",
    "gini":                      "shifting income distribution — inequality can widen or narrow rapidly",
    "poverty_headcount":         "unstable poverty levels — economic conditions shift for the most vulnerable",
    # Health
    "health_expenditure_gdp":    "inconsistent healthcare investment — access to public health services may be unreliable",
    "physicians_per_1000":       "uneven healthcare workforce — medical access varies year to year",
    "life_expectancy":           "fluctuating health outcomes — life expectancy can be affected by crises or policy changes",
    "infant_mortality":          "volatile infant health outcomes — public health system reliability is uncertain",
    # Education
    "education_spend_gdp":       "volatile education funding — quality of schooling may fluctuate significantly",
    # Infrastructure / access
    "sanitation_access":         "inconsistent sanitation — access to safe facilities is not guaranteed",
    "clean_water_access":        "unreliable water access — safe drinking water availability varies significantly",
    "electricity_access":        "unstable energy access — power availability is inconsistent for daily life and business",
    "internet_users_pct":        "uneven digital access — connectivity varies, affecting remote work and information access",
    "urban_population_pct":      "rapid urbanisation shifts — city capacity and quality can change fast",
    # Gender
    "women_in_parliament":       "fluctuating political gender representation — policy direction on gender equity is volatile",
    "adolescent_fertility_rate": "shifting demographic patterns — population age structure is in flux",
    # Environment
    "co2_per_capita":            "volatile emissions trajectory — environmental policy and industrial output fluctuate",
    "temp_anomaly":              "high climate variability — conditions are becoming less predictable",
    # Quality of life
    "teleport_score":            "inconsistent quality-of-life signals — composite scores shift with policy and economic changes",
}


# Which direction is "better" for each metric
_METRIC_HIGHER_IS_BETTER: dict[str, bool] = {
    # Higher = better
    "internet_users_pct":         True,
    "electricity_access":         True,
    "sanitation_access":          True,
    "clean_water_access":         True,
    "life_expectancy":            True,
    "gdp_per_capita":             True,
    "gni_per_capita_ppp":         True,
    "rule_of_law":                True,
    "political_stability":        True,
    "control_of_corruption":      True,
    "women_in_parliament":        True,
    "female_labor_participation": True,
    "teleport_score":             True,
    "physicians_per_1000":        True,
    "education_spend_gdp":        True,
    "health_expenditure_gdp":     True,
    "gdp_growth":                 True,
    "urban_population_pct":       True,
    # Lower = better
    "inflation":                  False,
    "homicide_rate":              False,
    "infant_mortality":           False,
    "poverty_headcount":          False,
    "conflict_events":            False,
    "fatalities":                 False,
    "co2_per_capita":             False,
    "gini":                       False,
    "unemployment":               False,
    "youth_unemployment":         False,
    "adolescent_fertility_rate":  False,
    "temp_anomaly":               False,
}


def _finding_best_worst(
    metric_name: str,
    stats_summary: dict,
    label: str,
    higher_is_better: bool,
) -> dict | None:
    """Generic best-vs-worst finding for any metric."""
    means = {
        c: stats_summary[c][metric_name]["mean"]
        for c in stats_summary
        if stats_summary[c].get(metric_name, {}).get("mean") is not None
    }
    if len(means) < 2:
        return None
    best  = max(means, key=lambda c: means[c]) if higher_is_better else min(means, key=lambda c: means[c])
    worst = min(means, key=lambda c: means[c]) if higher_is_better else max(means, key=lambda c: means[c])
    if best == worst:
        return None
    bv, wv = means[best], means[worst]
    gap = abs(bv - wv)
    direction_word = "highest" if higher_is_better else "lowest"
    return {
        "type":   "growth",
        "title":  f"{best} has the {direction_word} {label.lower()}",
        "value":  f"{bv:.2f}  (vs {wv:.2f} for {worst})",
        "detail": (
            f"{best} averages {bv:.2f} on {label.lower()} — "
            f"a gap of {gap:.2f} compared to {worst} ({wv:.2f}). "
            f"{'Higher' if higher_is_better else 'Lower'} {label.lower()} "
            f"{'is a positive signal' if higher_is_better else 'means better outcomes'} "
            f"for residents and migrants considering relocation."
        ),
    }


def _build_active_metrics(
    selected_tools: list[str],
    worldbank_indicators: list[str] | None = None,
) -> list[tuple[str, Any]]:
    """Return (metric_name, extractor) pairs ordered by tool priority.

    If worldbank_indicators is provided and non-empty, those metrics are
    prepended (in order) so they are treated as the most relevant.
    """
    seen: set[str] = set()
    result: list[tuple[str, Any]] = []

    # Prepend query-specific WB indicators first
    if worldbank_indicators:
        for metric in worldbank_indicators:
            if metric not in seen and metric in _ALL_METRICS:
                seen.add(metric)
                extractor, _ = _ALL_METRICS[metric]
                result.append((metric, extractor))

    # Then add remaining metrics from tool priority order
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


def _finding_anomaly(
    anomalies: dict,
    selected_tools: list[str],
    priority_override: list[str] | None = None,
) -> dict | None:
    """Surface the anomaly in the most query-relevant metric."""
    priority: list[str] = list(priority_override or [])
    for tool in selected_tools:
        for m in _TOOL_METRICS.get(tool, []):
            if m not in priority:
                priority.append(m)
    # Generic fallbacks
    for m in ["gdp_growth", "inflation", "unemployment", "temp_anomaly"]:
        if m not in priority:
            priority.append(m)

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


def _finding_governance_ranking(stats_summary: dict, selected_tools: list[str]) -> dict | None:
    """Best-vs-worst comparison for rule of law or control of corruption."""
    if "worldbank" not in selected_tools:
        return None
    for metric_key, label in [("rule_of_law", "rule of law"), ("control_of_corruption", "control of corruption")]:
        means = {
            c: stats_summary[c][metric_key]["mean"]
            for c in stats_summary
            if stats_summary[c].get(metric_key, {}).get("mean") is not None
        }
        if len(means) < 2:
            continue
        best  = max(means, key=lambda c: means[c])
        worst = min(means, key=lambda c: means[c])
        return {
            "type":   "growth",
            "title":  f"{best} leads on {label}",
            "value":  f"Index {means[best]:.2f}  (vs {means[worst]:.2f} for {worst})",
            "detail": (
                f"The World Bank {label} index places {best} at {means[best]:.2f} on average — "
                f"the highest score in this comparison group. {worst} scores {means[worst]:.2f}, "
                f"indicating weaker institutional quality and legal reliability. "
                f"For migrants, a higher {label} score signals predictable institutions and "
                f"consistent enforcement of rights and contracts."
            ),
        }
    return None


def _finding_life_expectancy(stats_summary: dict, selected_tools: list[str]) -> dict | None:
    """Highest vs lowest life expectancy comparison."""
    if "worldbank" not in selected_tools:
        return None
    means = {
        c: stats_summary[c]["life_expectancy"]["mean"]
        for c in stats_summary
        if stats_summary[c].get("life_expectancy", {}).get("mean") is not None
    }
    if len(means) < 2:
        return None
    best  = max(means, key=lambda c: means[c])
    worst = min(means, key=lambda c: means[c])
    gap   = means[best] - means[worst]
    return {
        "type":   "growth",
        "title":  f"{best} has the highest life expectancy",
        "value":  f"{means[best]:.1f} years  (vs {means[worst]:.1f} for {worst})",
        "detail": (
            f"{best}'s average life expectancy is {means[best]:.1f} years — "
            f"{gap:.1f} years more than {worst} ({means[worst]:.1f} years). "
            f"Life expectancy is a composite health signal that reflects healthcare system "
            f"quality, nutrition, and safety. For migrants, it is one of the clearest "
            f"indicators of the overall quality of life available in a country."
        ),
    }


def _finding_access_gap(stats_summary: dict, selected_tools: list[str]) -> dict | None:
    """Sanitation, clean water, or electricity access gap across countries."""
    if "worldbank" not in selected_tools:
        return None
    for metric_key, label in [
        ("sanitation_access", "sanitation access"),
        ("clean_water_access", "clean water access"),
        ("electricity_access", "electricity access"),
    ]:
        means = {
            c: stats_summary[c][metric_key]["mean"]
            for c in stats_summary
            if stats_summary[c].get(metric_key, {}).get("mean") is not None
        }
        if len(means) < 2:
            continue
        best  = max(means, key=lambda c: means[c])
        worst = min(means, key=lambda c: means[c])
        gap   = means[best] - means[worst]
        if gap < 5:
            continue  # not meaningful
        return {
            "type":   "volatility",
            "title":  f"Large gap in {label}",
            "value":  f"{means[best]:.1f}% vs {means[worst]:.1f}%",
            "detail": (
                f"{best} achieves {means[best]:.1f}% {label} — "
                f"{gap:.0f} percentage points higher than {worst} ({means[worst]:.1f}%). "
                f"This gap is a critical infrastructure signal: migrants moving from "
                f"low-access to high-access countries gain significantly improved "
                f"daily living conditions and public health outcomes."
            ),
        }
    return None


def _finding_volatility(
    stats_summary: dict,
    selected_tools: list[str],
    priority_metrics: list[str] | None = None,
) -> dict | None:
    """Most volatile metric in query-priority order."""
    priority: list[str] = list(priority_metrics or [])
    for tool in selected_tools:
        for m in _TOOL_METRICS.get(tool, []):
            if m not in priority:
                priority.append(m)

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
    query_focus: str = "",
    worldbank_indicators: list[str] | None = None,
) -> dict[str, Any]:
    """
    Query-aware EDA: metric set is driven by selected_tools and worldbank_indicators.

    worldbank_indicators (from LLM tool selector) are prepended so the most
    query-relevant metrics appear first and drive findings generation.

    Chart hints always prefer distribution_box and correlation_heatmap first —
    purely statistical visuals distinct from the time-series comparison charts.
    """
    findings: list[dict] = []
    growth_rates: dict[str, dict] = {}
    anomalies: dict[str, list] = {}
    stats_summary: dict[str, dict] = {}
    charts_hint: list[str] = []

    # ── Build tool-driven active metric list ──────────────────────────────────
    active_metrics = _build_active_metrics(selected_tools, worldbank_indicators)
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
    # Use the top worldbank_indicator as anchor if available
    _available_cols = set(latest_matrix.columns.tolist())
    if worldbank_indicators:
        anchor_metric = next(
            (m for m in worldbank_indicators if m in _available_cols),
            _primary_anchor(selected_tools, _available_cols),
        )
    else:
        anchor_metric = _primary_anchor(selected_tools, _available_cols)

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

    # ── 5. Narrative findings — query-topic priority order ────────────────────

    # Step 1: best/worst finding for top 3 most-relevant metrics
    for metric_name, _ in active_metrics[:3]:
        if len(findings) >= 2:
            break
        higher = _METRIC_HIGHER_IS_BETTER.get(metric_name, True)
        _, label = _ALL_METRICS.get(metric_name, (None, metric_name.replace("_", " ")))
        f = _finding_best_worst(metric_name, stats_summary, label, higher)
        if f:
            findings.append(f)
            charts_hint.append("distribution_box")

    # Step 2: CAGR trend for the top metric
    if len(findings) < 3 and active_metrics:
        top_metric = active_metrics[0][0]
        cagrs = {c: growth_rates[c].get(top_metric, {}).get("cagr") for c in growth_rates}
        cagrs = {c: v for c, v in cagrs.items() if v is not None}
        if len(cagrs) >= 2:
            higher = _METRIC_HIGHER_IS_BETTER.get(top_metric, True)
            leader = max(cagrs, key=lambda c: cagrs[c]) if higher else min(cagrs, key=lambda c: cagrs[c])
            lv = cagrs[leader]
            _, label = _ALL_METRICS.get(top_metric, (None, top_metric.replace("_", " ")))
            findings.append({
                "type":   "growth",
                "title":  f"{leader} shows {'fastest growth' if lv > 0 else 'sharpest decline'} in {label}",
                "value":  f"CAGR {lv * 100:+.1f}% per year",
                "detail": (
                    f"{leader}'s {label.lower()} is "
                    f"{'growing' if lv > 0 else 'shrinking'} {abs(lv) * 100:.1f}% per year — "
                    f"the {'strongest' if (higher and lv > 0) else 'most notable'} "
                    f"trajectory in this comparison."
                ),
            })
            charts_hint.append("growth_rate_bar")

    # Step 3: Anomaly (highest-z metric from active_metrics[:4])
    if len(findings) < 4:
        priority_override = [m for m, _ in active_metrics[:4]]
        f = _finding_anomaly(anomalies, selected_tools, priority_override=priority_override)
        if f:
            findings.append(f)
            charts_hint.append("anomaly_timeline")

    # Step 4: Correlation
    if len(findings) < 4:
        f = _finding_correlation(correlation_results, anchor_metric, len(countries_data))
        if f:
            findings.append(f)
            charts_hint.append("correlation_heatmap")

    # Step 5: Volatility fallback using active metric priority
    if len(findings) < 4:
        f = _finding_volatility(
            stats_summary,
            selected_tools,
            priority_metrics=[m for m, _ in active_metrics[:6]],
        )
        if f:
            findings.append(f)
            charts_hint.append("distribution_box")

    # Guarantee at least one statistical chart (not time-series)
    if "distribution_box" not in charts_hint and "correlation_heatmap" not in charts_hint:
        charts_hint = ["distribution_box"] + charts_hint

    return {
        "findings":              findings[:4],
        "growth_rates":          growth_rates,
        "anomalies":             anomalies,
        "stats_summary":         stats_summary,
        "correlations":          correlation_results[:6],
        "charts_hint":           list(dict.fromkeys(charts_hint))[:2],  # max 2, deduped
        "latest_matrix":         latest_matrix.to_dict() if not latest_matrix.empty else {},
        # Ordered list of metric names in query-priority order — used by chart builders
        # so EDA charts always show metrics relevant to the user's query, not generic ones.
        "active_metrics_ordered": [m for m, _ in active_metrics],
    }
