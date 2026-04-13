"""Plotly multi-panel charts for the migration workflow UI."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_migration_panels(dataset: dict[str, Any]) -> str:
    wb = pd.DataFrame(dataset.get("worldbank") or [])
    cl = pd.DataFrame(dataset.get("climate") or [])
    em = pd.DataFrame(dataset.get("employment") or [])
    disp = pd.DataFrame(dataset.get("displacement") or [])
    gd = pd.DataFrame(dataset.get("gdelt", {}).get("timeline") or [])

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Inflation vs displacement outflow",
            "Climate anomalies and precipitation",
            "GDELT media tone over time",
            "Employment trend",
        ),
    )

    if not wb.empty and {"year", "label", "value"}.issubset(wb.columns):
        infl = wb[wb["label"] == "inflation"].sort_values("year")
        if not infl.empty:
            fig.add_trace(
                go.Scatter(x=infl["year"], y=infl["value"], name="Inflation %", line={"color": "#b04a35"}),
                row=1,
                col=1,
            )
    if not disp.empty and {"year", "value"}.issubset(disp.columns):
        yearly_outflow = disp.groupby("year", as_index=False)["value"].sum().sort_values("year")
        if not yearly_outflow.empty:
            fig.add_trace(
                go.Bar(x=yearly_outflow["year"], y=yearly_outflow["value"], name="Displacement outflow", marker={"color": "#54707b"}, opacity=0.55),
                row=1,
                col=1,
            )

    if not cl.empty and "year" in cl.columns:
        climate = cl.sort_values("year")
        if "annual_precipitation_mm" in climate.columns:
            fig.add_trace(
                go.Bar(x=climate["year"], y=climate["annual_precipitation_mm"], name="Precip mm", marker={"color": "#6a78f0"}),
                row=1,
                col=2,
            )
        if "avg_temp_anomaly_c" in climate.columns:
            fig.add_trace(
                go.Scatter(x=climate["year"], y=climate["avg_temp_anomaly_c"], name="Temp anomaly C", line={"color": "#e06c4a"}),
                row=1,
                col=2,
            )

    if not gd.empty and "date" in gd.columns:
        gd = gd.copy()
        gd["date"] = pd.to_datetime(gd["date"], format="%Y%m%d", errors="coerce")
        gd = gd.dropna(subset=["date"])
        if "tone" in gd.columns and not gd.empty:
            tone = gd.groupby("date", as_index=False)["tone"].mean()
            fig.add_trace(
                go.Scatter(x=tone["date"], y=tone["tone"], name="GDELT tone", line={"color": "#2c8a7e"}),
                row=2,
                col=1,
            )

    if not em.empty and "year" in em.columns:
        em = em.sort_values("year")
        if "unemployment_rate" in em.columns:
            fig.add_trace(
                go.Scatter(x=em["year"], y=em["unemployment_rate"], name="Unemployment", line={"color": "#0f766e"}),
                row=2,
                col=2,
            )
        if "youth_unemployment_rate" in em.columns and em["youth_unemployment_rate"].notna().any():
            fig.add_trace(
                go.Scatter(x=em["year"], y=em["youth_unemployment_rate"], name="Youth unemployment", line={"color": "#cc5f1a"}),
                row=2,
                col=2,
            )

    fig.update_layout(
        height=720,
        title_text="Migration intelligence exploratory panels",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#fffdf8",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 32, "r": 20, "t": 72, "b": 28},
    )
    return fig.to_json()
