"""EDA: relocation_advisory — rank countries by weighted AQI + Teleport + WB proxies."""

from __future__ import annotations

import math
from collections import defaultdict

import pandas as pd

from models.schemas import IntentConfig, MigrationDataset, RelocationResult, RelocationRow


def _safe_norm(value: float, low: float, high: float, lower_better: bool = False) -> float:
    if math.isnan(value):
        return 0.0
    if high == low:
        return 0.5
    scaled = (value - low) / (high - low)
    return 1.0 - scaled if lower_better else scaled


def run_relocation_scorer(
    dataset: MigrationDataset,
    intent: IntentConfig,
) -> RelocationResult:
    if intent.intent != "relocation_advisory":
        return RelocationResult(active=False, narrative="Relocation scorer inactive for this intent.")

    aqi = pd.DataFrame(dataset.aqi or [])
    city = pd.DataFrame(dataset.city_scores or [])
    wb = pd.DataFrame(dataset.worldbank or [])

    # Check what data is available
    has_aqi = not aqi.empty and {"country", "pm25"}.issubset(aqi.columns)
    has_city = not city.empty and {"country", "category", "score_out_of_10"}.issubset(city.columns)
    has_wb = not wb.empty and {"country", "label", "value", "year"}.issubset(wb.columns)

    # If completely missing all three required sources, fail gracefully
    if not (has_aqi or has_city or has_wb):
        reasons = []
        if aqi.empty:
            reasons.append("AQI data")
        if city.empty:
            reasons.append("Teleport quality-of-life scores")
        if wb.empty:
            reasons.append("World Bank economic indicators")
        
        return RelocationResult(
            active=True,
            narrative=(
                f"Relocation scoring could not be completed. Key quality-of-life metrics, "
                f"including {', '.join(reasons)}, were missing."
            ),
        )

    weights = intent.weights or {}
    w_aqi = float(weights.get("aqi", 0.4))
    w_health = float(weights.get("healthcare", 0.35))
    w_edu = float(weights.get("education", 0.25))
    w_safety = float(weights.get("safety", 0.25))
    w_cost = float(weights.get("cost_of_living", 0.15))

    pm_by_country: dict[str, float] = {}
    if has_aqi:
        valid = aqi.dropna(subset=["country", "pm25"]).copy()
        if not valid.empty:
            pm_by_country = (
                valid.groupby("country", as_index=True)["pm25"].mean().astype(float).to_dict()
            )

    city_metrics: dict[str, dict[str, float]] = defaultdict(dict)
    if has_city:
        for _, row in city.dropna(subset=["country"]).iterrows():
            country = str(row["country"]).upper()[:3]
            category = str(row.get("category", "")).lower()
            score = float(row.get("score_out_of_10") or 0.0)
            if "health" in category:
                city_metrics[country]["health"] = score
            elif "education" in category:
                city_metrics[country]["education"] = score
            elif "safety" in category:
                city_metrics[country]["safety"] = score
            elif "cost of living" in category:
                city_metrics[country]["cost"] = score
            elif "economy" in category:
                city_metrics[country]["salary"] = score

    wb_metrics: dict[str, dict[str, float]] = defaultdict(dict)
    if has_wb:
        wb2 = wb.sort_values("year")
        latest = wb2.groupby(["country", "label"], as_index=False).tail(1)
        for _, row in latest.iterrows():
            wb_metrics[str(row["country"]).upper()[:3]][str(row["label"])] = float(row["value"])

    countries = sorted(set(pm_by_country) & set(city_metrics))
    if not countries:
        return RelocationResult(
            active=True,
            narrative=(
                "The relocation join produced no countries with both AQI and quality-of-life data. "
                "The system should report insufficient evidence instead of suggesting a country."
            ),
        )

    health_values = []
    edu_values = []
    safety_values = []
    cost_values = []
    salary_values = []
    pm_values = []
    for country in countries:
        pm = pm_by_country.get(country)
        if pm is not None:
            pm_values.append(pm)
        health_values.append(
            city_metrics[country].get("health")
            or wb_metrics[country].get("health_expenditure_gdp")
            or wb_metrics[country].get("physicians_per_1000")
            or 0.0
        )
        edu_values.append(
            city_metrics[country].get("education")
            or wb_metrics[country].get("education_spend_gdp")
            or 0.0
        )
        safety_values.append(city_metrics[country].get("safety") or 0.0)
        cost_values.append(city_metrics[country].get("cost") or 0.0)
        salary_values.append(
            city_metrics[country].get("salary")
            or wb_metrics[country].get("gdp_per_capita_usd")
            or 0.0
        )

    rows: list[RelocationRow] = []
    for country in countries:
        pm = float(pm_by_country.get(country) or 0.0)
        health = float(
            city_metrics[country].get("health")
            or wb_metrics[country].get("health_expenditure_gdp")
            or wb_metrics[country].get("physicians_per_1000")
            or 0.0
        )
        education = float(
            city_metrics[country].get("education")
            or wb_metrics[country].get("education_spend_gdp")
            or 0.0
        )
        safety = float(city_metrics[country].get("safety") or 0.0)
        cost = float(city_metrics[country].get("cost") or 0.0)
        salary = float(
            city_metrics[country].get("salary")
            or wb_metrics[country].get("gdp_per_capita_usd")
            or 0.0
        )

        score = (
            w_aqi * _safe_norm(pm, min(pm_values), max(pm_values), lower_better=True)
            + w_health * _safe_norm(health, min(health_values), max(health_values))
            + w_edu * _safe_norm(education, min(edu_values), max(edu_values))
            + w_safety * _safe_norm(safety, min(safety_values), max(safety_values))
            + w_cost * _safe_norm(cost, min(cost_values), max(cost_values), lower_better=True)
        )

        if salary_values and any(value > 0 for value in salary_values):
            score += 0.1 * _safe_norm(salary, min(salary_values), max(salary_values))

        rows.append(
            RelocationRow(
                country=country,
                weighted_score=round(score, 4),
                pm25=round(pm, 2),
                healthcare_signal=round(health, 2),
                education_signal=round(education, 2),
                safety_signal=round(safety, 2),
                cost_of_living_signal=round(cost, 2),
                salary_signal=round(salary, 2),
            )
        )

    rows.sort(key=lambda row: row.weighted_score, reverse=True)
    top = rows[:10]
    if len(top) < 3:
        return RelocationResult(
            active=True,
            top_countries=top,
            narrative=(
                "Only a very small candidate set satisfied the live AQI and relocation joins, "
                "so the system should treat the result as tentative."
            ),
        )
    return RelocationResult(
        active=True,
        top_countries=top,
        narrative=(
            "Countries ranked from live OpenAQ PM2.5, Teleport healthcare/education/safety/cost-of-living, "
            "and World Bank relocation indicators. Higher weighted scores reflect the user's stated priorities."
        ),
    )
