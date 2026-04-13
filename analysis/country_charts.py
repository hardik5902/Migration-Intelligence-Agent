"""Build 4 comparison charts across K countries for the country comparison pipeline."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go

COUNTRY_COLORS = [
    "#af3f2f",
    "#54707b",
    "#6a78f0",
    "#2d6a4f",
    "#e06c4a",
    "#5b4f9a",
    "#3d6b8a",
    "#c45c26",
]

_BASE_LAYOUT: dict[str, Any] = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#fffdf8",
    margin=dict(l=40, r=20, t=56, b=44),
    font=dict(family="Space Grotesk, sans-serif", size=12, color="#1e1b18"),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        x=0,
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=11),
    ),
    hoverlabel=dict(bgcolor="white", font_size=12),
)


def _color_map(countries: list[str]) -> dict[str, str]:
    return {c: COUNTRY_COLORS[i % len(COUNTRY_COLORS)] for i, c in enumerate(countries)}


def _apply_layout(
    fig: go.Figure,
    title: str,
    yaxis_title: str = "",
    height: int = 380,
) -> None:
    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(text=title, font=dict(size=14, color="#1e1b18"), x=0, pad=dict(l=0)),
        yaxis_title=yaxis_title,
        height=height,
        barmode="group",
    )
    fig.update_xaxes(gridcolor="rgba(30,27,24,0.06)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(30,27,24,0.06)", zeroline=False)


def _no_data(fig: go.Figure, label: str) -> None:
    fig.add_annotation(
        text=f"No {label} data available for the selected countries.",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=13, color="#6b6258"),
    )


# ---------------------------------------------------------------------------
# Chart 1: Economic — GDP per capita (worldbank) or best available
# ---------------------------------------------------------------------------

def build_economic_chart(countries_data: dict[str, Any], colors: dict[str, str]) -> str:
    """Line chart: GDP per capita over time per country."""
    fig = go.Figure()
    has_data = False

    for country, dataset in countries_data.items():
        wb = pd.DataFrame(dataset.get("worldbank") or [])
        if wb.empty or "label" not in wb.columns or "value" not in wb.columns:
            continue

        for label in ("gdp_per_capita_usd", "inflation", "health_expenditure_gdp"):
            series = wb[wb["label"] == label].dropna(subset=["value", "year"])
            if not series.empty:
                series = series.sort_values("year")
                fig.add_trace(
                    go.Scatter(
                        x=series["year"],
                        y=series["value"],
                        name=country,
                        mode="lines+markers",
                        line=dict(color=colors.get(country, "#af3f2f"), width=2.5),
                        marker=dict(size=5),
                    )
                )
                has_data = True
                break

    if not has_data:
        # Fallback: unemployment from employment tool
        for country, dataset in countries_data.items():
            emp = pd.DataFrame(dataset.get("employment") or [])
            if not emp.empty and "unemployment_rate" in emp.columns:
                emp = emp.dropna(subset=["unemployment_rate", "year"]).sort_values("year")
                if not emp.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=emp["year"],
                            y=emp["unemployment_rate"],
                            name=country,
                            mode="lines+markers",
                            line=dict(color=colors.get(country, "#af3f2f"), width=2.5),
                            marker=dict(size=5),
                        )
                    )
                    has_data = True

    if not has_data:
        _no_data(fig, "economic (World Bank)")

    _apply_layout(fig, "Economic: GDP per Capita Trend (USD)", "USD")
    return fig.to_json()


# ---------------------------------------------------------------------------
# Chart 2: Labor — Unemployment rate (employment) or poverty (worldbank)
# ---------------------------------------------------------------------------

def build_labor_chart(countries_data: dict[str, Any], colors: dict[str, str]) -> str:
    """Line chart: Unemployment rate over time per country."""
    fig = go.Figure()
    has_data = False

    for country, dataset in countries_data.items():
        emp = pd.DataFrame(dataset.get("employment") or [])
        if emp.empty or "unemployment_rate" not in emp.columns:
            continue
        emp = emp.dropna(subset=["unemployment_rate", "year"]).sort_values("year")
        if emp.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=emp["year"],
                y=emp["unemployment_rate"],
                name=country,
                mode="lines+markers",
                line=dict(color=colors.get(country, "#54707b"), width=2.5),
                marker=dict(size=5),
            )
        )
        has_data = True

    if not has_data:
        # Fallback: poverty headcount or inflation from worldbank
        for country, dataset in countries_data.items():
            wb = pd.DataFrame(dataset.get("worldbank") or [])
            if wb.empty or "label" not in wb.columns:
                continue
            for label in ("poverty_headcount", "inflation"):
                series = wb[wb["label"] == label].dropna(subset=["value", "year"])
                if not series.empty:
                    series = series.sort_values("year")
                    fig.add_trace(
                        go.Scatter(
                            x=series["year"],
                            y=series["value"],
                            name=country,
                            mode="lines+markers",
                            line=dict(color=colors.get(country, "#54707b"), width=2.5),
                            marker=dict(size=5),
                        )
                    )
                    has_data = True
                    break

    if not has_data:
        _no_data(fig, "labor market (ILO)")

    _apply_layout(fig, "Labor Market: Unemployment Rate (%)", "%")
    return fig.to_json()


# ---------------------------------------------------------------------------
# Chart 3: Environment — Average PM2.5 (aqi) or temperature anomaly (climate)
# ---------------------------------------------------------------------------

def build_environment_chart(countries_data: dict[str, Any], colors: dict[str, str]) -> str:
    """Bar chart: Average PM2.5 by country, with WHO guideline line."""
    fig = go.Figure()
    has_data = False

    aqi_vals: dict[str, float] = {}
    for country, dataset in countries_data.items():
        aqi = pd.DataFrame(dataset.get("aqi") or [])
        if not aqi.empty and "pm25" in aqi.columns:
            avg = aqi["pm25"].dropna().mean()
            if pd.notna(avg) and avg > 0:
                aqi_vals[country] = float(avg)

    if aqi_vals:
        sorted_vals = sorted(aqi_vals.items(), key=lambda x: x[1], reverse=True)
        fig.add_trace(
            go.Bar(
                x=[v[0] for v in sorted_vals],
                y=[v[1] for v in sorted_vals],
                marker_color=[colors.get(c, "#c45c26") for c, _ in sorted_vals],
                showlegend=False,
                hovertemplate="%{x}: %{y:.1f} μg/m³<extra></extra>",
            )
        )
        fig.add_hline(
            y=15,
            line_dash="dash",
            line_color="#6a78f0",
            annotation_text="WHO guideline 15 μg/m³",
            annotation_font_color="#6a78f0",
        )
        has_data = True
        _apply_layout(fig, "Air Quality: Average PM2.5 — lower is better (μg/m³)", "μg/m³")
    else:
        # Fallback: climate temperature anomaly
        climate_vals: dict[str, float] = {}
        for country, dataset in countries_data.items():
            clim = pd.DataFrame(dataset.get("climate") or [])
            if not clim.empty and "avg_temp_anomaly_c" in clim.columns:
                avg = clim["avg_temp_anomaly_c"].dropna().mean()
                if pd.notna(avg):
                    climate_vals[country] = float(avg)

        if climate_vals:
            sorted_vals = sorted(climate_vals.items(), key=lambda x: x[1], reverse=True)
            fig.add_trace(
                go.Bar(
                    x=[v[0] for v in sorted_vals],
                    y=[v[1] for v in sorted_vals],
                    marker_color=[colors.get(c, "#e06c4a") for c, _ in sorted_vals],
                    showlegend=False,
                    hovertemplate="%{x}: %{y:.2f}°C<extra></extra>",
                )
            )
            has_data = True
            _apply_layout(fig, "Climate: Average Temperature Anomaly (°C)", "°C")
        else:
            _no_data(fig, "environmental (AQI/Climate)")
            _apply_layout(fig, "Environment Data", "")

    return fig.to_json()


# ---------------------------------------------------------------------------
# Chart 4: Migration/Safety — Displacement (UNHCR), conflict (ACLED),
#          or quality-of-life (Teleport), whichever has data
# ---------------------------------------------------------------------------

def build_migration_chart(countries_data: dict[str, Any], colors: dict[str, str]) -> str:
    """Bar chart: UNHCR displacement, ACLED conflict events, or Teleport QoL score."""
    fig = go.Figure()
    has_data = False

    # Priority 1: UNHCR displacement
    disp_vals: dict[str, float] = {}
    for country, dataset in countries_data.items():
        disp = pd.DataFrame(dataset.get("displacement") or [])
        if not disp.empty and "value" in disp.columns:
            total = float(disp["value"].sum())
            if total > 0:
                disp_vals[country] = total

    if disp_vals:
        sorted_vals = sorted(disp_vals.items(), key=lambda x: x[1], reverse=True)
        fig.add_trace(
            go.Bar(
                x=[v[0] for v in sorted_vals],
                y=[v[1] for v in sorted_vals],
                marker_color=[colors.get(c, "#5b4f9a") for c, _ in sorted_vals],
                showlegend=False,
                hovertemplate="%{x}: %{y:,.0f} persons<extra></extra>",
            )
        )
        has_data = True
        _apply_layout(fig, "UNHCR: Total Refugee Displacement Outflow", "Persons")
    else:
        # Priority 2: ACLED conflict events
        conf_vals: dict[str, int] = {}
        for country, dataset in countries_data.items():
            conf = pd.DataFrame(dataset.get("conflict_events") or [])
            if not conf.empty:
                conf_vals[country] = len(conf)

        if conf_vals:
            sorted_vals = sorted(conf_vals.items(), key=lambda x: x[1], reverse=True)
            fig.add_trace(
                go.Bar(
                    x=[v[0] for v in sorted_vals],
                    y=[v[1] for v in sorted_vals],
                    marker_color=[colors.get(c, "#8b2942") for c, _ in sorted_vals],
                    showlegend=False,
                    hovertemplate="%{x}: %{y} events<extra></extra>",
                )
            )
            has_data = True
            _apply_layout(fig, "ACLED: Conflict Events Count", "Events")
        else:
            # Priority 3: Teleport quality-of-life
            qol_vals: dict[str, float] = {}
            for country, dataset in countries_data.items():
                cs = pd.DataFrame(dataset.get("city_scores") or [])
                if not cs.empty and "score_out_of_10" in cs.columns:
                    avg = cs["score_out_of_10"].dropna().mean()
                    if pd.notna(avg):
                        qol_vals[country] = float(avg)

            if qol_vals:
                sorted_vals = sorted(qol_vals.items(), key=lambda x: x[1], reverse=True)
                fig.add_trace(
                    go.Bar(
                        x=[v[0] for v in sorted_vals],
                        y=[v[1] for v in sorted_vals],
                        marker_color=[colors.get(c, "#2d6a4f") for c, _ in sorted_vals],
                        showlegend=False,
                        hovertemplate="%{x}: %{y:.1f}/10<extra></extra>",
                    )
                )
                has_data = True
                _apply_layout(fig, "Teleport: Quality-of-Life Score (0–10)", "Score /10")
            else:
                _no_data(fig, "migration/safety (UNHCR/ACLED/Teleport)")
                _apply_layout(fig, "Migration & Safety Data", "")

    return fig.to_json()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_country_comparison_charts(
    countries_data: dict[str, Any],
    selected_tools: list[str],
    query_focus: str,
) -> list[str]:
    """Build exactly 4 comparison charts. Returns list of 4 Plotly JSON strings."""
    countries = list(countries_data.keys())
    colors = _color_map(countries)

    return [
        build_economic_chart(countries_data, colors),
        build_labor_chart(countries_data, colors),
        build_environment_chart(countries_data, colors),
        build_migration_chart(countries_data, colors),
    ]
