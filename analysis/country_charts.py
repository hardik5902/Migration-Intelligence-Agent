"""Build 4 comparison charts across K countries.

Uses a priority-ordered metric registry so charts always show the best available
data — no hardcoded label assumptions that break when the API returns different labels.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go

# ── Colour palette ─────────────────────────────────────────────────────────────
COUNTRY_COLORS = [
    "#af3f2f", "#54707b", "#6a78f0", "#2d6a4f",
    "#e06c4a", "#5b4f9a", "#3d6b8a", "#c45c26",
]

# ── Base layout shared by all charts ──────────────────────────────────────────
def _base_layout(title: str, yaxis_title: str, source: str, height: int = 460) -> dict:
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,253,248,0.9)",
        height=height,
        autosize=True,
        margin=dict(l=72, r=32, t=58, b=130),
        font=dict(family="Space Grotesk, sans-serif", size=12, color="#1e1b18"),
        title=dict(
            text=title,
            font=dict(size=13, color="#1e1b18", family="Source Serif 4, serif"),
            x=0,
            xanchor="left",
            pad=dict(l=4, b=6),
        ),
        yaxis=dict(
            title=yaxis_title,
            title_standoff=12,
            automargin=True,
            gridcolor="rgba(30,27,24,0.07)",
            zerolinecolor="rgba(30,27,24,0.15)",
            tickfont=dict(size=11),
        ),
        xaxis=dict(
            automargin=True,
            gridcolor="rgba(30,27,24,0.07)",
            zerolinecolor="rgba(30,27,24,0.15)",
            tickfont=dict(size=11),
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,253,248,0.85)",
            bordercolor="rgba(30,27,24,0.1)",
            borderwidth=1,
            font=dict(size=11),
        ),
        hoverlabel=dict(bgcolor="white", font_size=12, bordercolor="#e0d8cc"),
        annotations=[
            dict(
                text=f"Source: {source}",
                xref="paper", yref="paper",
                x=1, y=-0.38,
                xanchor="right", yanchor="top",
                showarrow=False,
                font=dict(size=9, color="#9b8f84"),
            )
        ],
        barmode="group",
    )


def _color_map(countries: list[str]) -> dict[str, str]:
    return {c: COUNTRY_COLORS[i % len(COUNTRY_COLORS)] for i, c in enumerate(countries)}


def _no_data_fig(title: str, message: str, source: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=12, color="#9b8f84"),
        bgcolor="rgba(255,250,241,0.8)",
        bordercolor="#e0d8cc",
        borderwidth=1,
        borderpad=10,
    )
    fig.update_layout(**_base_layout(title, "", source))
    return fig


# ── Extraction helpers ─────────────────────────────────────────────────────────

def _wb_series(dataset: dict, *labels: str) -> pd.DataFrame | None:
    """Return a year/value DataFrame for the first matching worldbank label."""
    wb = pd.DataFrame(dataset.get("worldbank") or [])
    if wb.empty or "label" not in wb.columns or "value" not in wb.columns:
        return None
    for lbl in labels:
        sub = wb[wb["label"] == lbl].dropna(subset=["value", "year"])
        if not sub.empty:
            return sub[["year", "value"]].sort_values("year")
    return None


def _emp_series(dataset: dict, col: str) -> pd.DataFrame | None:
    emp = pd.DataFrame(dataset.get("employment") or [])
    if emp.empty or col not in emp.columns:
        return None
    sub = emp.dropna(subset=[col, "year"])
    if sub.empty:
        return None
    return sub[["year", col]].rename(columns={col: "value"}).sort_values("year")


def _climate_series(dataset: dict, col: str) -> pd.DataFrame | None:
    clim = pd.DataFrame(dataset.get("climate") or [])
    if clim.empty or col not in clim.columns:
        return None
    sub = clim.dropna(subset=[col, "year"])
    if sub.empty:
        return None
    return sub[["year", col]].rename(columns={col: "value"}).sort_values("year")


def _scalar_gdp_per_capita(dataset: dict) -> float | None:
    series = _wb_series(dataset, "gdp_per_capita_usd")
    if series is None or series.empty:
        return None
    last = series.dropna(subset=["value"]).sort_values("year").iloc[-1]
    return float(last["value"])


def _scalar_unemployment(dataset: dict) -> float | None:
    emp = pd.DataFrame(dataset.get("employment") or [])
    if emp.empty or "unemployment_rate" not in emp.columns:
        return None
    sub = emp.dropna(subset=["unemployment_rate", "year"])
    if sub.empty:
        return None
    return float(sub.sort_values("year").iloc[-1]["unemployment_rate"])


def _scalar_gdp_growth(dataset: dict) -> float | None:
    series = _wb_series(dataset, "gdp_growth")
    if series is None or series.empty:
        return None
    return float(series.sort_values("year").iloc[-1]["value"])


def _scalar_inflation(dataset: dict) -> float | None:
    series = _wb_series(dataset, "inflation")
    if series is None or series.empty:
        return None
    return float(series.sort_values("year").iloc[-1]["value"])


def _scalar_aqi(dataset: dict) -> float | None:
    aqi = pd.DataFrame(dataset.get("aqi") or [])
    if aqi.empty or "pm25" not in aqi.columns:
        return None
    # Filter out corrupt readings (negative or impossibly high)
    valid = aqi["pm25"].dropna()
    valid = valid[(valid > 0) & (valid <= 1000)]
    if valid.empty:
        return None
    v = valid.mean()
    return float(v) if pd.notna(v) else None


def _scalar_disp(dataset: dict) -> float | None:
    disp = pd.DataFrame(dataset.get("displacement") or [])
    if disp.empty or "value" not in disp.columns:
        return None
    total = disp["value"].sum()
    return float(total) if total > 0 else None


def _scalar_conflict(dataset: dict) -> float | None:
    conf = pd.DataFrame(dataset.get("conflict_events") or [])
    return float(len(conf)) if not conf.empty else None


def _scalar_teleport(dataset: dict) -> float | None:
    cs = pd.DataFrame(dataset.get("city_scores") or [])
    if cs.empty or "score_out_of_10" not in cs.columns:
        return None
    v = cs["score_out_of_10"].dropna().mean()
    return float(v) if pd.notna(v) else None


# ── Line chart builder (time series per country) ───────────────────────────────

def _build_line_chart(
    countries_data: dict,
    colors: dict,
    extractor,
    title: str,
    yaxis_title: str,
    source: str,
) -> go.Figure:
    fig = go.Figure()
    has_data = False

    for country, dataset in countries_data.items():
        series = extractor(dataset)
        if series is None or series.empty:
            continue
        fig.add_trace(go.Scatter(
            x=series["year"],
            y=series["value"],
            name=country,
            mode="lines+markers",
            line=dict(color=colors.get(country, "#af3f2f"), width=2.5),
            marker=dict(size=6, symbol="circle"),
            hovertemplate=f"<b>{country}</b><br>Year: %{{x}}<br>{yaxis_title}: %{{y:.2f}}<extra></extra>",
        ))
        has_data = True

    if not has_data:
        return _no_data_fig(
            title,
            f"No {source} data returned for the selected countries.<br>"
            "Try adding this tool to your query or expanding the year range.",
            source,
        )

    fig.update_layout(**_base_layout(title, yaxis_title, source))
    return fig


# ── Bar chart builder (one bar per country, snapshot value) ────────────────────

def _build_bar_chart(
    countries_data: dict,
    colors: dict,
    extractor,
    title: str,
    yaxis_title: str,
    source: str,
    reference_y: float | None = None,
    reference_label: str = "",
    ascending: bool = True,
) -> go.Figure:
    vals: dict[str, float] = {}
    for country, dataset in countries_data.items():
        v = extractor(dataset)
        if v is not None:
            vals[country] = v

    if not vals:
        return _no_data_fig(
            title,
            f"No {source} data returned for the selected countries.",
            source,
        )

    sorted_items = sorted(vals.items(), key=lambda x: x[1], reverse=not ascending)
    countries_sorted = [k for k, _ in sorted_items]
    values_sorted = [v for _, v in sorted_items]

    fig = go.Figure(go.Bar(
        x=countries_sorted,
        y=values_sorted,
        marker_color=[colors.get(c, "#af3f2f") for c in countries_sorted],
        hovertemplate="<b>%{x}</b><br>" + yaxis_title + ": %{y:,.2f}<extra></extra>",
        showlegend=False,
    ))

    if reference_y is not None:
        fig.add_hline(
            y=reference_y,
            line_dash="dash",
            line_color="#6a78f0",
            line_width=1.5,
            annotation_text=reference_label,
            annotation_font=dict(color="#6a78f0", size=10),
            annotation_position="top right",
        )

    fig.update_layout(**_base_layout(title, yaxis_title, source))
    return fig


# ── Scatter chart builder (country positioning: x vs y, one point per country) ─

def _build_scatter_chart(
    countries_data: dict,
    colors: dict,
    x_extractor,
    y_extractor,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    source: str,
) -> go.Figure:
    x_vals: dict[str, float] = {}
    y_vals: dict[str, float] = {}
    for country, dataset in countries_data.items():
        xv = x_extractor(dataset)
        yv = y_extractor(dataset)
        if xv is not None and yv is not None:
            x_vals[country] = xv
            y_vals[country] = yv

    common = [c for c in x_vals if c in y_vals]
    if len(common) < 2:
        return _no_data_fig(title, f"Need ≥ 2 countries with both {xaxis_title} and {yaxis_title}.", source)

    fig = go.Figure()
    for country in common:
        col = colors.get(country, "#af3f2f")
        fig.add_trace(go.Scatter(
            x=[x_vals[country]],
            y=[y_vals[country]],
            name=country,
            mode="markers+text",
            text=[country],
            textposition="top center",
            textfont=dict(size=10),
            marker=dict(size=14, color=col, line=dict(color="white", width=1.5)),
            hovertemplate=(
                f"<b>{country}</b><br>"
                f"{xaxis_title}: %{{x:,.1f}}<br>"
                f"{yaxis_title}: %{{y:.2f}}<extra></extra>"
            ),
        ))

    layout = _base_layout(title, xaxis_title, source)
    layout["xaxis"]["title"] = xaxis_title
    layout["yaxis"]["title"] = yaxis_title
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig


# ── Area chart builder (filled line per country over time) ─────────────────────

def _build_area_chart(
    countries_data: dict,
    colors: dict,
    extractor,
    title: str,
    yaxis_title: str,
    source: str,
) -> go.Figure:
    fig = go.Figure()
    has_data = False

    for country, dataset in countries_data.items():
        series = extractor(dataset)
        if series is None or series.empty:
            continue
        col = colors.get(country, "#af3f2f")
        # Convert hex to rgba for fill
        r, g, b = int(col[1:3], 16), int(col[3:5], 16), int(col[5:7], 16)
        fig.add_trace(go.Scatter(
            x=series["year"],
            y=series["value"],
            name=country,
            mode="lines",
            line=dict(color=col, width=2.5),
            fill="tozeroy",
            fillcolor=f"rgba({r},{g},{b},0.12)",
            hovertemplate=f"<b>{country}</b><br>Year: %{{x}}<br>{yaxis_title}: %{{y:.2f}}<extra></extra>",
        ))
        has_data = True

    if not has_data:
        return _no_data_fig(title, f"No {source} data for the selected countries.", source)

    fig.update_layout(**_base_layout(title, yaxis_title, source))
    return fig


# ── Metric registry: ordered by desirability ──────────────────────────────────
#
# Each entry defines:
#   type       – "line" or "bar"
#   extractor  – callable(dataset) → DataFrame | float | None
#   title, yaxis_title, source
#   optional:  reference_y, reference_label (for bar charts)

def _metric_registry(countries_data: dict) -> list[dict]:
    return [
        # ── Economic (line) ────────────────────────────────────────────────────
        {
            "type": "line",
            "extractor": lambda d: _wb_series(d, "gdp_growth"),
            "title": "GDP Growth Rate (%)",
            "yaxis_title": "Annual %",
            "source": "World Bank",
        },
        # ── Scatter: country positioning (worldbank + employment) ─────────────
        {
            "type": "scatter",
            "x_extractor": _scalar_gdp_per_capita,
            "y_extractor": _scalar_unemployment,
            "title": "Country Positioning: GDP per Capita vs Unemployment Rate",
            "xaxis_title": "GDP per Capita (USD)",
            "yaxis_title": "Unemployment Rate (%)",
            "source": "World Bank / ILO",
        },
        # ── Economic (line) ────────────────────────────────────────────────────
        {
            "type": "line",
            "extractor": lambda d: _wb_series(d, "inflation"),
            "title": "Consumer Price Inflation (%)",
            "yaxis_title": "CPI Annual %",
            "source": "World Bank",
        },
        # ── Labour (line) ──────────────────────────────────────────────────────
        {
            "type": "line",
            "extractor": lambda d: _emp_series(d, "unemployment_rate"),
            "title": "Unemployment Rate (%)",
            "yaxis_title": "% of labour force",
            "source": "ILO / World Bank",
        },
        # ── Area: GDP per capita trajectory (worldbank only) ─────────────────
        {
            "type": "area",
            "extractor": lambda d: _wb_series(d, "gdp_per_capita_usd"),
            "title": "GDP per Capita Trajectory — Filled Area (USD)",
            "yaxis_title": "GDP per Capita (USD)",
            "source": "World Bank",
        },
        # ── Scatter: inflation vs unemployment (worldbank + employment) ───────
        {
            "type": "scatter",
            "x_extractor": _scalar_inflation,
            "y_extractor": _scalar_unemployment,
            "title": "Macroeconomic Mix: Inflation vs Unemployment (latest)",
            "xaxis_title": "Inflation (%)",
            "yaxis_title": "Unemployment Rate (%)",
            "source": "World Bank / ILO",
        },
        # ── Bar: snapshot GDP per capita (worldbank only) ─────────────────────
        {
            "type": "bar",
            "extractor": _scalar_gdp_per_capita,
            "title": "GDP per Capita — Latest Snapshot (USD)",
            "yaxis_title": "USD per capita",
            "source": "World Bank",
            "ascending": False,
        },
        {
            "type": "line",
            "extractor": lambda d: _wb_series(d, "unemployment"),
            "title": "Unemployment Rate – World Bank (%)",
            "yaxis_title": "%",
            "source": "World Bank",
        },
        {
            "type": "line",
            "extractor": lambda d: _emp_series(d, "youth_unemployment_rate"),
            "title": "Youth Unemployment Rate (%)",
            "yaxis_title": "% youth labour",
            "source": "ILO / World Bank",
        },
        # ── Area: inflation trend (worldbank only) ────────────────────────────
        {
            "type": "area",
            "extractor": lambda d: _wb_series(d, "inflation"),
            "title": "Inflation Trend — Filled Area (%)",
            "yaxis_title": "CPI Annual %",
            "source": "World Bank",
        },
        # ── Inequality / governance ────────────────────────────────────────────
        {
            "type": "line",
            "extractor": lambda d: _wb_series(d, "gini"),
            "title": "Income Inequality (Gini Index)",
            "yaxis_title": "Gini (0–100)",
            "source": "World Bank",
        },
        {
            "type": "line",
            "extractor": lambda d: _wb_series(d, "political_stability"),
            "title": "Political Stability Index",
            "yaxis_title": "Score",
            "source": "World Bank WGI",
        },
        # ── Scatter: GDP per capita vs GDP growth (worldbank only) ────────────
        {
            "type": "scatter",
            "x_extractor": _scalar_gdp_per_capita,
            "y_extractor": _scalar_gdp_growth,
            "title": "Wealth vs Momentum: GDP per Capita vs GDP Growth",
            "xaxis_title": "GDP per Capita (USD)",
            "yaxis_title": "GDP Growth (%)",
            "source": "World Bank",
        },
        # ── Area: unemployment trend (employment only) ────────────────────────
        {
            "type": "area",
            "extractor": lambda d: _emp_series(d, "unemployment_rate"),
            "title": "Unemployment Trend — Filled Area (%)",
            "yaxis_title": "% of labour force",
            "source": "ILO / World Bank",
        },
        # ── Climate ────────────────────────────────────────────────────────────
        {
            "type": "line",
            "extractor": lambda d: _climate_series(d, "avg_temp_anomaly_c"),
            "title": "Temperature Anomaly vs Baseline (°C)",
            "yaxis_title": "°C",
            "source": "Open-Meteo",
        },
        {
            "type": "line",
            "extractor": lambda d: _climate_series(d, "annual_precipitation_mm"),
            "title": "Annual Precipitation (mm)",
            "yaxis_title": "mm/year",
            "source": "Open-Meteo",
        },
        # ── Environment snapshot (bar) ─────────────────────────────────────────
        {
            "type": "bar",
            "extractor": _scalar_aqi,
            "title": "Air Quality: Average PM2.5 — lower is better",
            "yaxis_title": "μg/m³",
            "source": "OpenAQ / World Bank",
            "reference_y": 15.0,
            "reference_label": "WHO 15 μg/m³",
            "ascending": False,
        },
        # ── Migration / safety snapshot (bar) ─────────────────────────────────
        {
            "type": "bar",
            "extractor": _scalar_disp,
            "title": "UNHCR Refugee Displacement Outflow",
            "yaxis_title": "Persons",
            "source": "UNHCR",
            "ascending": False,
        },
        {
            "type": "bar",
            "extractor": _scalar_conflict,
            "title": "ACLED: Conflict Event Count",
            "yaxis_title": "Events",
            "source": "ACLED",
            "ascending": False,
        },
        {
            "type": "bar",
            "extractor": _scalar_teleport,
            "title": "Quality-of-Life Score (Teleport, 0–10)",
            "yaxis_title": "Score /10",
            "source": "Teleport",
            "ascending": True,
        },
        # ── Area: displacement outflow ──────────────────────────────────────────
        {
            "type": "area",
            "extractor": lambda d: (
                (lambda disp:
                    disp.groupby("year", as_index=False)["value"]
                    .sum().sort_values("year")
                    if not disp.empty and {"year", "value"}.issubset(disp.columns)
                    else None
                )(pd.DataFrame(d.get("displacement") or []))
            ),
            "title": "Refugee Displacement Outflow over Time (filled area)",
            "yaxis_title": "Persons displaced",
            "source": "UNHCR",
        },
        # ── Area: temperature anomaly ───────────────────────────────────────────
        {
            "type": "area",
            "extractor": lambda d: _climate_series(d, "avg_temp_anomaly_c"),
            "title": "Temperature Anomaly Trend — Filled Area (°C)",
            "yaxis_title": "°C vs baseline",
            "source": "Open-Meteo",
        },
    ]


def _count_countries_with_data(metric: dict, countries_data: dict) -> int:
    """How many countries have non-None data for this metric."""
    mtype = metric.get("type", "line")

    if mtype == "scatter":
        # Both x and y must be present
        x_ext = metric.get("x_extractor")
        y_ext = metric.get("y_extractor")
        if not x_ext or not y_ext:
            return 0
        count = 0
        for dataset in countries_data.values():
            try:
                if x_ext(dataset) is not None and y_ext(dataset) is not None:
                    count += 1
            except Exception:
                pass
        return count

    extractor = metric.get("extractor")
    if not extractor:
        return 0
    count = 0
    for dataset in countries_data.values():
        try:
            result = extractor(dataset)
            if result is not None:
                if isinstance(result, pd.DataFrame):
                    if not result.empty:
                        count += 1
                else:
                    count += 1
        except Exception:
            pass
    return count


# ── Public entry point ─────────────────────────────────────────────────────────

def build_country_comparison_charts(
    countries_data: dict[str, Any],
    selected_tools: list[str],
    query_focus: str,
) -> list[str]:
    """
    Build exactly 4 Plotly charts.

    Dynamically picks the 4 metrics with the most country coverage from the
    priority-ordered registry, then falls back to empty-state charts if needed.
    """
    countries = list(countries_data.keys())
    colors = _color_map(countries)

    # Score each metric by how many countries have data and keep only
    # candidates that actually have data — avoids padding with blanks.
    registry = _metric_registry(countries_data)
    candidates = [
        (m, _count_countries_with_data(m, countries_data))
        for m in registry
    ]
    candidates = [(m, n) for m, n in candidates if n >= 2]

    # Two-pass selection:
    #   Pass 1 (variety): at most 1 of each visual type — max 4 distinct types.
    #   Pass 2 (relaxed): cap each type at 2 to fill remaining slots.
    #   Pass 3 (final fill): any leftover candidates, no type cap.
    seen_titles: set[str] = set()
    chosen: list[dict] = []

    def _pick(max_per_type: int | None) -> None:
        type_counts: dict[str, int] = {}
        for metric in chosen:
            t = metric.get("type", "line")
            type_counts[t] = type_counts.get(t, 0) + 1
        for metric, _n in candidates:
            if len(chosen) >= 4:
                return
            key = metric["title"]
            if key in seen_titles:
                continue
            mtype = metric.get("type", "line")
            if max_per_type is not None and type_counts.get(mtype, 0) >= max_per_type:
                continue
            seen_titles.add(key)
            type_counts[mtype] = type_counts.get(mtype, 0) + 1
            chosen.append(metric)

    _pick(max_per_type=1)   # variety first
    _pick(max_per_type=2)   # allow pairs
    _pick(max_per_type=None)  # fill rest

    # If we still couldn't fill 4, fall back to empty-state placeholders
    while len(chosen) < 4:
        chosen.append({
            "type": "empty",
            "title": "No Additional Data",
            "message": "No further data was returned by the selected tools.",
            "source": "",
        })

    # Build figures
    charts: list[str] = []
    for metric in chosen[:4]:
        mtype = metric.get("type", "line")
        try:
            if mtype == "empty":
                fig = _no_data_fig(metric["title"], metric.get("message", ""), metric["source"])
            elif mtype == "line":
                fig = _build_line_chart(
                    countries_data, colors,
                    metric["extractor"],
                    metric["title"],
                    metric["yaxis_title"],
                    metric["source"],
                )
            elif mtype == "bar":
                fig = _build_bar_chart(
                    countries_data, colors,
                    metric["extractor"],
                    metric["title"],
                    metric["yaxis_title"],
                    metric["source"],
                    reference_y=metric.get("reference_y"),
                    reference_label=metric.get("reference_label", ""),
                    ascending=metric.get("ascending", False),
                )
            elif mtype == "scatter":
                fig = _build_scatter_chart(
                    countries_data, colors,
                    metric["x_extractor"],
                    metric["y_extractor"],
                    metric["title"],
                    metric["xaxis_title"],
                    metric["yaxis_title"],
                    metric["source"],
                )
            elif mtype == "area":
                fig = _build_area_chart(
                    countries_data, colors,
                    metric["extractor"],
                    metric["title"],
                    metric["yaxis_title"],
                    metric["source"],
                )
            else:
                fig = _no_data_fig(metric["title"], "Unknown chart type.", metric.get("source", ""))
        except Exception:
            fig = _no_data_fig(
                metric.get("title", "Chart"),
                "An error occurred while building this chart.",
                metric.get("source", ""),
            )
        charts.append(fig.to_json())

    return charts
