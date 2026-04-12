"""EDA: relocation_advisory — rank countries by weighted AQI + Teleport + WB proxies."""

from __future__ import annotations

import math
from collections import defaultdict

import pandas as pd

from models.schemas import IntentConfig, MigrationDataset, RelocationResult, RelocationRow


def run_relocation_scorer(
    dataset: MigrationDataset,
    intent: IntentConfig,
) -> RelocationResult:
    if intent.intent != "relocation_advisory":
        return RelocationResult(active=False, narrative="Relocation scorer inactive for this intent.")

    aqi = pd.DataFrame(dataset.aqi or [])
    cs = pd.DataFrame(dataset.city_scores or [])

    w = intent.weights or {}
    w_aqi = float(w.get("aqi", 0.4))
    w_health = float(w.get("healthcare", 0.35))
    w_edu = float(w.get("education", 0.25))

    pm_by_country: dict[str, list[float]] = defaultdict(list)
    if not aqi.empty and "country" in aqi.columns:
        for _, r in aqi.iterrows():
            pm = r.get("pm25")
            if pm is None or (isinstance(pm, float) and math.isnan(pm)):
                continue
            pm_by_country[str(r["country"])].append(float(pm))

    health_edu: dict[str, dict[str, float]] = defaultdict(dict)
    for _, r in cs.iterrows():
        cat = str(r.get("category", "")).lower()
        slug = str(r.get("slug", "unknown"))
        if "health" in cat:
            health_edu[slug]["health"] = float(r.get("score_out_of_10") or 0)
        if "education" in cat:
            health_edu[slug]["edu"] = float(r.get("score_out_of_10") or 0)

    # Map slug to country heuristic: use slug prefix
    slug_country = {k: k.upper()[:3] for k in health_edu}

    rows: list[RelocationRow] = []
    countries = sorted(pm_by_country.keys())
    if not countries:
        return RelocationResult(active=True, narrative="No PM2.5 readings returned from OpenAQ for this query.")

    def norm(xs: list[float], v: float, lower_better: bool) -> float:
        lo, hi = min(xs), max(xs)
        if hi == lo:
            return 0.5
        t = (v - lo) / (hi - lo)
        return 1.0 - t if lower_better else t

    pm_country_mean = {c: sum(v) / len(v) for c, v in pm_by_country.items()}
    pm_vals = list(pm_country_mean.values())

    for c in countries:
        pm = pm_country_mean[c]
        n_pm = norm(pm_vals, pm, lower_better=True)
        # pick best health/edu slug heuristically matching country code in slug
        h, e = 5.0, 5.0
        for slug, scores in health_edu.items():
            if c.lower() in slug or slug.startswith(c.lower()):
                h = scores.get("health", h)
                e = scores.get("edu", e)
        n_h = h / 10.0
        n_e = e / 10.0
        score = w_aqi * n_pm + w_health * n_h + w_edu * n_e
        rows.append(
            RelocationRow(
                country=c,
                weighted_score=round(score, 4),
                pm25=round(pm, 2),
                healthcare_signal=h,
                education_signal=e,
            )
        )

    rows.sort(key=lambda r: r.weighted_score, reverse=True)
    top = rows[:10]
    return RelocationResult(
        active=True,
        top_countries=top,
        narrative="Ranked by weighted AQI (lower PM2.5 better), Teleport health/education scores, and user weights.",
    )
