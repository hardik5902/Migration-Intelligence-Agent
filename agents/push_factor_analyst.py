"""EDA: correlate macro / climate / conflict proxies vs displacement outflow."""

from __future__ import annotations

from typing import Any

import pandas as pd

from analysis.correlation import run_correlation_analysis, run_granger_test
from models.schemas import IntentConfig, MigrationDataset, PushFactor, PushFactorResult


def run_push_factor_analysis(
    dataset: MigrationDataset,
    intent: IntentConfig,
) -> PushFactorResult:
    disp = pd.DataFrame(dataset.displacement or [])
    wb = pd.DataFrame(dataset.worldbank or [])
    cl = pd.DataFrame(dataset.climate or [])
    conf = pd.DataFrame(dataset.conflict_events or [])

    if disp.empty or "year" not in disp.columns:
        return PushFactorResult(narrative="Insufficient displacement time series for correlation.")

    outflow = (
        disp.groupby("year", as_index=False)["value"]
        .sum()
        .rename(columns={"value": "outflow"})
    )

    wide_parts: list[pd.DataFrame] = [outflow]
    if not wb.empty and {"year", "label", "value"}.issubset(wb.columns):
        piv = wb.pivot_table(index="year", columns="label", values="value", aggfunc="mean")
        piv.columns = [f"wb_{c}" for c in piv.columns]
        piv = piv.reset_index()
        wide_parts.append(piv)
    if not cl.empty and {"year", "annual_precipitation_mm"}.issubset(cl.columns):
        c2 = cl.groupby("year", as_index=False)["annual_precipitation_mm"].mean()
        wide_parts.append(c2)
    if not conf.empty and "event_date" in conf.columns:
        conf = conf.copy()
        conf["year"] = pd.to_datetime(conf["event_date"], errors="coerce").dt.year
        c3 = conf.groupby("year", as_index=False)["fatalities"].sum().rename(
            columns={"fatalities": "conflict_fatalities"}
        )
        wide_parts.append(c3)

    merged = wide_parts[0]
    for p in wide_parts[1:]:
        merged = merged.merge(p, on="year", how="outer")
    merged = merged.sort_values("year").dropna(subset=["outflow"], how="all")
    if merged.shape[0] < 4:
        return PushFactorResult(narrative="Too few overlapping years after merge.")

    merged = merged.ffill().bfill()
    data = merged.to_dict(orient="list")
    ranked = run_correlation_analysis(data, "outflow")
    top_name = ranked[0]["indicator"] if ranked else ""
    top_r = ranked[0]["pearson_r"] if ranked else None

    gr: dict = {"lag_years": None, "p_value": None, "is_causal": False}
    if top_name and top_name in merged.columns:
        yv = merged["outflow"].astype(float).tolist()
        gr = run_granger_test(merged[top_name].astype(float).tolist(), yv)

    factors = [
        PushFactor(
            name=r["indicator"],
            pearson_r=r["pearson_r"],
            p_value=r["pearson_p"],
        )
        for r in ranked[:8]
    ]

    return PushFactorResult(
        top_driver=top_name,
        r=top_r,
        granger_lag_years=gr.get("lag_years"),
        p_value=gr.get("p_value"),
        ranked_factors=factors,
        narrative=(
            f"Strongest linear association with annual displacement totals: {top_name or 'n/a'}"
            + (f" (r={top_r:.3f})." if top_r is not None else ".")
        ),
    )
