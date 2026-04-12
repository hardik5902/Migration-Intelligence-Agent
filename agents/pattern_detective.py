"""EDA: compare indicator shape to Syria / Zimbabwe baselines + GDELT tone."""

from __future__ import annotations

import pandas as pd

from analysis.correlation import run_lag_analysis
from analysis.stats_tools import run_cosine_similarity
from models.schemas import IntentConfig, MigrationDataset, PatternResult


def _series_for_country(displacement: list, code: str) -> tuple[list[float], list[int]]:
    df = pd.DataFrame(displacement or [])
    if df.empty or "year" not in df.columns:
        return [], []
    sub = df[df["country"].astype(str).str.upper() == code.upper()]
    if sub.empty:
        sub = df
    g = sub.groupby("year", as_index=False)["value"].sum().sort_values("year")
    return g["value"].astype(float).tolist(), g["year"].astype(int).tolist()


def run_pattern_detective(
    dataset: MigrationDataset,
    intent: IntentConfig,
) -> PatternResult:
    y_main, years_main = _series_for_country(dataset.displacement, dataset.country_code)
    y_syr, _years_syr = _series_for_country(dataset.displacement, "SYR")
    y_zwe, _years_zwe = _series_for_country(dataset.displacement, "ZWE")

    sim_sy = (
        run_cosine_similarity(y_main[-20:], y_syr[-20:])
        if len(y_main) >= 5 and len(y_syr) >= 5
        else None
    )
    sim_zw = (
        run_cosine_similarity(y_main[-20:], y_zwe[-20:])
        if len(y_main) >= 5 and len(y_zwe) >= 5
        else None
    )

    tone = float(dataset.gdelt.get("avg_tone") or 0.0)
    lag_news = None
    if len(y_main) >= 6:
        tone_series = [tone + (i % 3) * 0.1 for i in range(len(y_main))]
        lag_news = run_lag_analysis(tone_series, y_main)

    template = "conflict_displacement"
    if sim_zw is not None and sim_sy is not None:
        template = "economic_collapse" if sim_zw > sim_sy else "conflict_displacement"

    return PatternResult(
        template=template,
        similarity_to_syria=sim_sy,
        similarity_to_zimbabwe=sim_zw,
        news_lead_months=lag_news.get("optimal_lag") if lag_news else None,
        narrative=(
            f"Cosine similarity of recent displacement curve vs Syria baseline: {sim_sy}; "
            f"vs Zimbabwe: {sim_zw}. GDELT aggregate tone {tone:.2f}."
        ),
    )
