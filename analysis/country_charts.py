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
def _base_layout(title: str, yaxis_title: str, source: str, height: int = 420) -> dict:
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,253,248,0.9)",
        height=height,
        autosize=True,
        margin=dict(l=64, r=24, t=52, b=90),
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
            y=-0.18,
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
                x=1, y=-0.28,
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
            "ascending": True,  # higher = better, show ascending (highest first)
        },
    ]


def _count_countries_with_data(metric: dict, countries_data: dict) -> int:
    """How many countries have non-None data for this metric."""
    extractor = metric["extractor"]
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

    # Score each metric by how many countries have data
    registry = _metric_registry(countries_data)
    scored = [(m, _count_countries_with_data(m, countries_data)) for m in registry]

    # Deduplicate: keep first occurrence of each (title, source) pair
    seen_titles: set[str] = set()
    chosen: list[dict] = []
    for metric, n_countries in scored:
        key = metric["title"]
        if key in seen_titles:
            continue
        if n_countries > 0:
            seen_titles.add(key)
            chosen.append(metric)
        if len(chosen) == 4:
            break

    # Pad to 4 with empty-state placeholders if necessary
    placeholders = [
        ("No Additional Data", "No further data was returned by the selected tools.", ""),
    ]
    while len(chosen) < 4:
        ph = placeholders[0]
        chosen.append({
            "type": "empty",
            "title": ph[0],
            "message": ph[1],
            "source": ph[2],
        })

    # Build figures
    charts: list[str] = []
    for metric in chosen[:4]:
        if metric["type"] == "empty":
            fig = _no_data_fig(metric["title"], metric.get("message", ""), metric["source"])
        elif metric["type"] == "line":
            fig = _build_line_chart(
                countries_data, colors,
                metric["extractor"],
                metric["title"],
                metric["yaxis_title"],
                metric["source"],
            )
        else:  # bar
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
        charts.append(fig.to_json())

    return charts
