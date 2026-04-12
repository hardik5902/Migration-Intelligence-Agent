"""Plotly multi-panel charts for Streamlit."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_migration_panels(dataset: dict[str, Any]) -> str:
    """Four-panel figure JSON: inflation vs outflow proxy, climate, tone, unemployment."""
    wb = pd.DataFrame(dataset.get("worldbank") or [])
    cl = pd.DataFrame(dataset.get("climate") or [])
    em = pd.DataFrame(dataset.get("employment") or [])
    gd = dataset.get("gdelt") or {}

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Macro indicators (World Bank)",
            "Climate (annual precip / °C max)",
            "News tone (GDELT aggregate)",
            "Unemployment (%)",
        ),
    )

    if not wb.empty and "year" in wb.columns:
        infl = wb[wb.get("label", pd.Series()) == "inflation"] if "label" in wb.columns else wb
        gdp = wb[wb["label"] == "gdp_growth"] if "label" in wb.columns else wb
        if not infl.empty:
            fig.add_trace(
                go.Scatter(x=infl["year"], y=infl["value"], name="Inflation %"),
                row=1,
                col=1,
            )
        if not gdp.empty:
            fig.add_trace(
                go.Scatter(x=gdp["year"], y=gdp["value"], name="GDP growth %"),
                row=1,
                col=1,
            )

    if not cl.empty and "year" in cl.columns:
        fig.add_trace(
            go.Bar(x=cl["year"], y=cl["annual_precipitation_mm"], name="Precip mm"),
            row=1,
            col=2,
        )

    tone = float(gd.get("avg_tone") or 0)
    fig.add_trace(go.Bar(x=["avg_tone"], y=[tone], name="GDELT tone"), row=2, col=1)

    if not em.empty and "year" in em.columns:
        fig.add_trace(
            go.Scatter(x=em["year"], y=em["unemployment_rate"], name="Unemployment"),
            row=2,
            col=2,
        )

    fig.update_layout(height=640, showlegend=True, title_text="Migration intelligence panels")
    return fig.to_json()
