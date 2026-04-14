"""Build 4 comparison charts across K countries.

Uses a priority-ordered metric registry so charts always show the best available
data — no hardcoded label assumptions that break when the API returns different labels.
"""

from __future__ import annotations

import re
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


def _scalar_life_expectancy(dataset: dict) -> float | None:
    series = _wb_series(dataset, "life_expectancy")
    if series is None or series.empty:
        return None
    return float(series.sort_values("year").iloc[-1]["value"])


def _scalar_infant_mortality(dataset: dict) -> float | None:
    series = _wb_series(dataset, "infant_mortality")
    if series is None or series.empty:
        return None
    return float(series.sort_values("year").iloc[-1]["value"])


def _scalar_sanitation_access(dataset: dict) -> float | None:
    series = _wb_series(dataset, "sanitation_access")
    if series is None or series.empty:
        return None
    return float(series.sort_values("year").iloc[-1]["value"])


def _scalar_clean_water_access(dataset: dict) -> float | None:
    series = _wb_series(dataset, "clean_water_access")
    if series is None or series.empty:
        return None
    return float(series.sort_values("year").iloc[-1]["value"])


def _scalar_electricity_access(dataset: dict) -> float | None:
    series = _wb_series(dataset, "electricity_access")
    if series is None or series.empty:
        return None
    return float(series.sort_values("year").iloc[-1]["value"])


def _scalar_internet_users(dataset: dict) -> float | None:
    series = _wb_series(dataset, "internet_users_pct")
    if series is None or series.empty:
        return None
    return float(series.sort_values("year").iloc[-1]["value"])


def _scalar_homicide_rate(dataset: dict) -> float | None:
    series = _wb_series(dataset, "homicide_rate")
    if series is None or series.empty:
        return None
    return float(series.sort_values("year").iloc[-1]["value"])


def _scalar_women_parliament(dataset: dict) -> float | None:
    series = _wb_series(dataset, "women_in_parliament")
    if series is None or series.empty:
        return None
    return float(series.sort_values("year").iloc[-1]["value"])


def _scalar_rule_of_law(dataset: dict) -> float | None:
    series = _wb_series(dataset, "rule_of_law")
    if series is None or series.empty:
        return None
    return float(series.sort_values("year").iloc[-1]["value"])


def _scalar_co2_per_capita(dataset: dict) -> float | None:
    series = _wb_series(dataset, "co2_per_capita")
    if series is None or series.empty:
        return None
    return float(series.sort_values("year").iloc[-1]["value"])


def _scalar_gni_per_capita(dataset: dict) -> float | None:
    series = _wb_series(dataset, "gni_per_capita_ppp")
    if series is None or series.empty:
        return None
    return float(series.sort_values("year").iloc[-1]["value"])


def _scalar_health_expenditure(dataset: dict) -> float | None:
    series = _wb_series(dataset, "health_expenditure_gdp")
    if series is None or series.empty:
        return None
    return float(series.sort_values("year").iloc[-1]["value"])


def _scalar_physicians(dataset: dict) -> float | None:
    series = _wb_series(dataset, "physicians_per_1000")
    if series is None or series.empty:
        return None
    return float(series.sort_values("year").iloc[-1]["value"])


def _scalar_education(dataset: dict) -> float | None:
    series = _wb_series(dataset, "education_spend_gdp")
    if series is None or series.empty:
        return None
    return float(series.sort_values("year").iloc[-1]["value"])


def _scalar_poverty(dataset: dict) -> float | None:
    series = _wb_series(dataset, "poverty_headcount")
    if series is None or series.empty:
        return None
    return float(series.sort_values("year").iloc[-1]["value"])


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


# ── Query-relevance scoring ────────────────────────────────────────────────────
#
# Each registry entry carries a `tags` list.  _topic_score() counts how many
# distinct tags have a keyword match inside the query_focus string.  Candidates
# are sorted by (score DESC, coverage DESC) before the variety-pick passes, so
# the 4 chosen charts always reflect what the user actually asked about.

_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "income":            ["income", "gdp", "salary", "wages", "earnings", "wealth",
                          "rich", "prosperity", "afford", "purchasing power", "standard of living"],
    "economic_stability":["economic", "stability", "stable", "macro", "economy",
                          "financial", "fiscal", "recession", "growth"],
    "employment":        ["employment", "job", "work", "unemployment", "labour",
                          "labor", "workforce", "career", "opportunity", "hire", "occupation"],
    "youth":             ["youth", "young", "graduate", "student", "millennial"],
    "education":         ["education", "school", "university", "college", "learning",
                          "academic", "study", "literacy", "knowledge", "training", "skill"],
    "health":            ["health", "healthcare", "medical", "hospital", "doctor",
                          "physician", "wellness", "medicine", "disease", "care"],
    "longevity":         ["longevity", "life expectancy", "lifespan", "age", "aging",
                          "mortality", "survival"],
    "child":             ["child", "children", "infant", "baby", "maternal", "newborn",
                          "birth", "pediatric"],
    "safety":            ["safe", "safety", "conflict", "violence", "war", "crime",
                          "terrorism", "security", "danger", "risk", "threat", "peace"],
    "governance":        ["governance", "political", "corruption", "democracy",
                          "rule of law", "institution", "policy", "government", "law"],
    "environment":       ["environment", "environmental", "climate", "carbon",
                          "emissions", "green", "sustainable", "nature", "ecology", "co2"],
    "air_quality":       ["air", "pollution", "pm2.5", "smog", "breathe", "air quality"],
    "infrastructure":    ["infrastructure", "sanitation", "electricity", "power",
                          "utility", "public service", "access", "facility"],
    "water":             ["water", "sanitation", "sewage", "drinking", "clean water",
                          "hygiene", "wash"],
    "internet":          ["internet", "digital", "technology", "connectivity", "online",
                          "tech", "innovation", "startup", "broadband"],
    "poverty":           ["poverty", "poor", "destitute", "deprivation", "developing",
                          "underdeveloped", "welfare", "subsistence"],
    "inequality":        ["inequality", "gini", "gap", "disparity", "income gap",
                          "wealth gap", "equal"],
    "gender":            ["gender", "women", "female", "equality", "girl",
                          "maternity", "feminist", "lgbtq", "diversity"],
    "quality_of_life":   ["quality of life", "quality", "lifestyle", "living standard",
                          "wellbeing", "happiness", "relocation", "migrate", "moving",
                          "expat", "emigrate", "immigrate", "settle"],
    "cost_of_living":    ["cost of living", "affordable", "cheap", "expensive",
                          "price level", "inflation", "purchasing"],
}


def _topic_score(metric: dict, query_lower: str) -> int:
    """Count how many distinct topic tags from the entry match the query string."""
    if not query_lower:
        return 0
    score = 0
    for tag in metric.get("tags", []):
        for kw in _TOPIC_KEYWORDS.get(tag, [tag]):
            if kw in query_lower:
                score += 1
                break  # one keyword match per tag is sufficient
    return score


# ── Metric registry ────────────────────────────────────────────────────────────
#
# Each entry defines:
#   type       – "line", "bar", "area", or "scatter"
#   tags       – topic list used by _topic_score() for query-relevance ranking
#   extractor  – callable(dataset) → DataFrame | float | None
#   title, yaxis_title, source
#   optional:  data_key (dedup), reference_y, reference_label, ascending

def _metric_registry(countries_data: dict) -> list[dict]:
    return [
        # ── Economy ────────────────────────────────────────────────────────────
        {
            "type": "line",
            "tags": ["economic_stability", "income", "growth"],
            "extractor": lambda d: _wb_series(d, "gdp_growth"),
            "title": "GDP Growth Rate (%)",
            "yaxis_title": "Annual %",
            "source": "World Bank",
        },
        {
            "type": "scatter",
            "tags": ["income", "employment"],
            "x_extractor": _scalar_gdp_per_capita,
            "y_extractor": _scalar_unemployment,
            "title": "Country Positioning: GDP per Capita vs Unemployment Rate",
            "xaxis_title": "GDP per Capita (USD)",
            "yaxis_title": "Unemployment Rate (%)",
            "source": "World Bank / ILO",
        },
        {
            "type": "line",
            "tags": ["economic_stability", "cost_of_living"],
            "extractor": lambda d: _wb_series(d, "inflation"),
            "title": "Consumer Price Inflation (%)",
            "yaxis_title": "CPI Annual %",
            "source": "World Bank",
        },
        {
            "type": "line",
            "tags": ["employment"],
            "data_key": "unemployment_rate",
            "extractor": lambda d: _emp_series(d, "unemployment_rate"),
            "title": "Unemployment Rate (%)",
            "yaxis_title": "% of labour force",
            "source": "ILO / World Bank",
        },
        {
            "type": "area",
            "tags": ["income", "wealth"],
            "data_key": "gdp_per_capita",
            "extractor": lambda d: _wb_series(d, "gdp_per_capita_usd"),
            "title": "GDP per Capita Trajectory — Filled Area (USD)",
            "yaxis_title": "GDP per Capita (USD)",
            "source": "World Bank",
        },
        {
            "type": "scatter",
            "tags": ["economic_stability", "employment", "cost_of_living"],
            "x_extractor": _scalar_inflation,
            "y_extractor": _scalar_unemployment,
            "title": "Macroeconomic Mix: Inflation vs Unemployment (latest)",
            "xaxis_title": "Inflation (%)",
            "yaxis_title": "Unemployment Rate (%)",
            "source": "World Bank / ILO",
        },
        {
            "type": "bar",
            "tags": ["income", "wealth"],
            "data_key": "gdp_per_capita",
            "extractor": _scalar_gdp_per_capita,
            "title": "GDP per Capita — Latest Snapshot (USD)",
            "yaxis_title": "USD per capita",
            "source": "World Bank",
            "ascending": False,
        },
        {
            "type": "line",
            "tags": ["employment"],
            "extractor": lambda d: _wb_series(d, "unemployment"),
            "title": "Unemployment Rate – World Bank (%)",
            "yaxis_title": "%",
            "source": "World Bank",
        },
        {
            "type": "line",
            "tags": ["employment", "youth"],
            "extractor": lambda d: _emp_series(d, "youth_unemployment_rate"),
            "title": "Youth Unemployment Rate (%)",
            "yaxis_title": "% youth labour",
            "source": "ILO / World Bank",
        },
        # ── Inequality / governance ────────────────────────────────────────────
        {
            "type": "line",
            "tags": ["inequality", "poverty"],
            "extractor": lambda d: _wb_series(d, "gini"),
            "title": "Income Inequality (Gini Index)",
            "yaxis_title": "Gini (0–100)",
            "source": "World Bank",
        },
        {
            "type": "line",
            "tags": ["safety", "governance"],
            "extractor": lambda d: _wb_series(d, "political_stability"),
            "title": "Political Stability Index",
            "yaxis_title": "Score",
            "source": "World Bank WGI",
        },
        # ── Health ─────────────────────────────────────────────────────────────
        {
            "type": "line",
            "tags": ["health"],
            "data_key": "health_expenditure_gdp",
            "extractor": lambda d: _wb_series(d, "health_expenditure_gdp"),
            "title": "Health Expenditure (% of GDP)",
            "yaxis_title": "% of GDP",
            "source": "World Bank",
        },
        {
            "type": "bar",
            "tags": ["health"],
            "data_key": "health_expenditure_gdp",
            "extractor": _scalar_health_expenditure,
            "title": "Health Expenditure — Latest Snapshot (% GDP)",
            "yaxis_title": "% of GDP",
            "source": "World Bank",
            "ascending": False,
        },
        {
            "type": "bar",
            "tags": ["health"],
            "data_key": "physicians_per_1000",
            "extractor": _scalar_physicians,
            "title": "Physicians per 1,000 People",
            "yaxis_title": "Physicians / 1,000",
            "source": "World Bank",
            "ascending": False,
        },
        # ── Education ──────────────────────────────────────────────────────────
        {
            "type": "line",
            "tags": ["education"],
            "data_key": "education_spend_gdp",
            "extractor": lambda d: _wb_series(d, "education_spend_gdp"),
            "title": "Education Spending (% of GDP)",
            "yaxis_title": "% of GDP",
            "source": "World Bank",
        },
        {
            "type": "bar",
            "tags": ["education"],
            "data_key": "education_spend_gdp",
            "extractor": _scalar_education,
            "title": "Education Spending — Latest Snapshot (% GDP)",
            "yaxis_title": "% of GDP",
            "source": "World Bank",
            "ascending": False,
        },
        # ── Poverty ────────────────────────────────────────────────────────────
        {
            "type": "bar",
            "tags": ["poverty", "inequality"],
            "data_key": "poverty_headcount",
            "extractor": _scalar_poverty,
            "title": "Poverty Headcount — National Poverty Line (%)",
            "yaxis_title": "% of population",
            "source": "World Bank",
            "ascending": False,
        },
        # ── Health scatters ────────────────────────────────────────────────────
        {
            "type": "scatter",
            "tags": ["health"],
            "x_extractor": _scalar_health_expenditure,
            "y_extractor": _scalar_physicians,
            "title": "Healthcare Investment: Spending vs Physician Density",
            "xaxis_title": "Health Expenditure (% GDP)",
            "yaxis_title": "Physicians per 1,000",
            "source": "World Bank",
        },
        {
            "type": "scatter",
            "tags": ["poverty", "income", "inequality"],
            "x_extractor": _scalar_gdp_per_capita,
            "y_extractor": _scalar_poverty,
            "title": "Wealth vs Poverty: GDP per Capita vs Poverty Rate",
            "xaxis_title": "GDP per Capita (USD)",
            "yaxis_title": "Poverty Headcount (%)",
            "source": "World Bank",
        },
        {
            "type": "scatter",
            "tags": ["income", "economic_stability", "growth"],
            "x_extractor": _scalar_gdp_per_capita,
            "y_extractor": _scalar_gdp_growth,
            "title": "Wealth vs Momentum: GDP per Capita vs GDP Growth",
            "xaxis_title": "GDP per Capita (USD)",
            "yaxis_title": "GDP Growth (%)",
            "source": "World Bank",
        },
        # ── Life expectancy (line + bar) ──────────────────────────────────────
        {
            "type": "line",
            "tags": ["health", "longevity"],
            "data_key": "life_expectancy",
            "extractor": lambda d: _wb_series(d, "life_expectancy"),
            "title": "Life Expectancy at Birth (years)",
            "yaxis_title": "Years",
            "source": "World Bank",
        },
        {
            "type": "bar",
            "tags": ["health", "longevity"],
            "data_key": "life_expectancy_bar",
            "extractor": _scalar_life_expectancy,
            "title": "Life Expectancy — Latest Snapshot (years)",
            "yaxis_title": "Years",
            "source": "World Bank",
            "ascending": False,
        },
        # ── Infant mortality (bar) ─────────────────────────────────────────────
        {
            "type": "bar",
            "tags": ["health", "child"],
            "data_key": "infant_mortality",
            "extractor": _scalar_infant_mortality,
            "title": "Infant Mortality Rate (per 1,000 live births) — lower is better",
            "yaxis_title": "Deaths per 1,000",
            "source": "World Bank",
            "ascending": True,
        },
        # ── Scatter: life expectancy vs health expenditure ────────────────────
        {
            "type": "scatter",
            "tags": ["health", "longevity"],
            "x_extractor": _scalar_health_expenditure,
            "y_extractor": _scalar_life_expectancy,
            "title": "Health Investment vs Outcomes: Spending vs Life Expectancy",
            "xaxis_title": "Health Expenditure (% GDP)",
            "yaxis_title": "Life Expectancy (years)",
            "source": "World Bank",
        },
        # ── Sanitation & clean water (bar) ─────────────────────────────────────
        {
            "type": "bar",
            "tags": ["infrastructure", "water"],
            "data_key": "sanitation_access",
            "extractor": _scalar_sanitation_access,
            "title": "Sanitation Access (% of population)",
            "yaxis_title": "% population",
            "source": "World Bank",
            "ascending": False,
        },
        {
            "type": "bar",
            "tags": ["water", "infrastructure"],
            "data_key": "clean_water_access",
            "extractor": _scalar_clean_water_access,
            "title": "Clean Water Access (% of population)",
            "yaxis_title": "% population",
            "source": "World Bank",
            "ascending": False,
        },
        # ── Scatter: sanitation vs clean water (infrastructure comparison) ────
        {
            "type": "scatter",
            "tags": ["water", "infrastructure"],
            "x_extractor": _scalar_sanitation_access,
            "y_extractor": _scalar_clean_water_access,
            "title": "Infrastructure Gap: Sanitation vs Clean Water Access",
            "xaxis_title": "Sanitation Access (%)",
            "yaxis_title": "Clean Water Access (%)",
            "source": "World Bank",
        },
        # ── Electricity & internet access ────────────────────────────────────
        {
            "type": "bar",
            "tags": ["infrastructure", "internet"],
            "data_key": "electricity_access",
            "extractor": _scalar_electricity_access,
            "title": "Electricity Access (% of population)",
            "yaxis_title": "% population",
            "source": "World Bank",
            "ascending": False,
        },
        {
            "type": "line",
            "tags": ["internet", "infrastructure"],
            "data_key": "internet_users",
            "extractor": lambda d: _wb_series(d, "internet_users_pct"),
            "title": "Internet Users (% of population)",
            "yaxis_title": "% population",
            "source": "World Bank",
        },
        {
            "type": "bar",
            "tags": ["internet", "infrastructure"],
            "data_key": "internet_users_bar",
            "extractor": _scalar_internet_users,
            "title": "Internet Penetration — Latest Snapshot (%)",
            "yaxis_title": "% population",
            "source": "World Bank",
            "ascending": False,
        },
        # ── Homicide rate (bar) ──────────────────────────────────────────────
        {
            "type": "bar",
            "tags": ["safety", "governance"],
            "data_key": "homicide_rate",
            "extractor": _scalar_homicide_rate,
            "title": "Homicide Rate (per 100,000 people) — lower is safer",
            "yaxis_title": "Homicides per 100k",
            "source": "World Bank",
            "ascending": True,
        },
        # ── Rule of law (line) ───────────────────────────────────────────────
        {
            "type": "line",
            "tags": ["governance", "safety"],
            "data_key": "rule_of_law",
            "extractor": lambda d: _wb_series(d, "rule_of_law"),
            "title": "Rule of Law Index",
            "yaxis_title": "Score (WGI)",
            "source": "World Bank WGI",
        },
        # ── Scatter: rule of law vs homicide rate ─────────────────────────────
        {
            "type": "scatter",
            "tags": ["governance", "safety"],
            "x_extractor": _scalar_rule_of_law,
            "y_extractor": _scalar_homicide_rate,
            "title": "Governance vs Safety: Rule of Law vs Homicide Rate",
            "xaxis_title": "Rule of Law Index",
            "yaxis_title": "Homicide Rate (per 100k)",
            "source": "World Bank",
        },
        # ── Women in parliament (bar + line) ─────────────────────────────────
        {
            "type": "bar",
            "tags": ["gender", "governance"],
            "data_key": "women_in_parliament",
            "extractor": _scalar_women_parliament,
            "title": "Women in Parliament (% of seats)",
            "yaxis_title": "% of seats",
            "source": "World Bank",
            "ascending": False,
        },
        {
            "type": "line",
            "tags": ["gender", "governance"],
            "data_key": "women_parliament_trend",
            "extractor": lambda d: _wb_series(d, "women_in_parliament"),
            "title": "Women in Parliament — Trend (%)",
            "yaxis_title": "% of seats",
            "source": "World Bank",
        },
        # ── CO₂ per capita (line + bar) ──────────────────────────────────────
        {
            "type": "line",
            "tags": ["environment"],
            "data_key": "co2_trend",
            "extractor": lambda d: _wb_series(d, "co2_per_capita"),
            "title": "CO₂ Emissions per Capita (tonnes)",
            "yaxis_title": "Tonnes CO₂/person",
            "source": "World Bank",
        },
        {
            "type": "bar",
            "tags": ["environment"],
            "data_key": "co2_bar",
            "extractor": _scalar_co2_per_capita,
            "title": "CO₂ Emissions — Latest Snapshot (t per capita)",
            "yaxis_title": "Tonnes CO₂/person",
            "source": "World Bank",
            "ascending": False,
        },
        # ── GNI per capita PPP (bar) ─────────────────────────────────────────
        {
            "type": "bar",
            "tags": ["income", "quality_of_life"],
            "data_key": "gni_per_capita",
            "extractor": _scalar_gni_per_capita,
            "title": "GNI per Capita PPP — Purchasing Power (int. $)",
            "yaxis_title": "Int. $/person",
            "source": "World Bank",
            "ascending": False,
        },
        # ── Scatter: GNI per capita vs life expectancy ────────────────────────
        {
            "type": "scatter",
            "tags": ["income", "longevity", "quality_of_life"],
            "x_extractor": _scalar_gni_per_capita,
            "y_extractor": _scalar_life_expectancy,
            "title": "Living Standards: GNI per Capita vs Life Expectancy",
            "xaxis_title": "GNI per Capita PPP (int. $)",
            "yaxis_title": "Life Expectancy (years)",
            "source": "World Bank",
        },
        # ── Climate ────────────────────────────────────────────────────────────
        {
            "type": "line",
            "tags": ["environment"],
            "extractor": lambda d: _climate_series(d, "avg_temp_anomaly_c"),
            "title": "Temperature Anomaly vs Baseline (°C)",
            "yaxis_title": "°C",
            "source": "Open-Meteo",
        },
        {
            "type": "line",
            "tags": ["environment"],
            "extractor": lambda d: _climate_series(d, "annual_precipitation_mm"),
            "title": "Annual Precipitation (mm)",
            "yaxis_title": "mm/year",
            "source": "Open-Meteo",
        },
        # ── Environment snapshot (bar) ─────────────────────────────────────────
        {
            "type": "bar",
            "tags": ["air_quality", "environment"],
            "extractor": _scalar_aqi,
            "title": "Air Quality: Average PM2.5 — lower is better",
            "yaxis_title": "μg/m³",
            "source": "OpenAQ / World Bank",
            "reference_y": 15.0,
            "reference_label": "WHO 15 μg/m³",
            "ascending": False,
        },
        # ── Safety snapshot (bar) ─────────────────────────────────────────────
        {
            "type": "bar",
            "tags": ["safety"],
            "extractor": _scalar_conflict,
            "title": "ACLED: Conflict Event Count",
            "yaxis_title": "Events",
            "source": "ACLED",
            "ascending": False,
        },
        {
            "type": "bar",
            "tags": ["quality_of_life"],
            "extractor": _scalar_teleport,
            "title": "Quality-of-Life Score (Teleport, 0–10)",
            "yaxis_title": "Score /10",
            "source": "Teleport",
            "ascending": True,
        },
        # ── Area: temperature anomaly ───────────────────────────────────────────
        {
            "type": "area",
            "tags": ["environment"],
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


# ── Registry key helpers ───────────────────────────────────────────────────────

def _entry_key(metric: dict, idx: int) -> str:
    """Derive a stable string key for a registry entry.

    Prefers the explicit `data_key` field; otherwise slugifies the title.
    The visual type is always appended to distinguish line vs bar for the same
    underlying metric (e.g. `life_expectancy_line` vs `life_expectancy_bar`).
    """
    if metric.get("data_key"):
        return f"{metric['data_key']}_{metric['type']}"
    slug = re.sub(r"[^a-z0-9]+", "_", metric.get("title", "").lower()).strip("_")
    return f"{slug[:40]}_{metric['type']}"


def _countries_with_data(metric: dict, countries_data: dict) -> list[str]:
    """Return names of countries that have non-None data for this metric."""
    mtype = metric.get("type", "line")
    result: list[str] = []

    if mtype == "scatter":
        x_ext = metric.get("x_extractor")
        y_ext = metric.get("y_extractor")
        if not x_ext or not y_ext:
            return result
        for country, dataset in countries_data.items():
            try:
                if x_ext(dataset) is not None and y_ext(dataset) is not None:
                    result.append(country)
            except Exception:
                pass
        return result

    extractor = metric.get("extractor")
    if not extractor:
        return result
    for country, dataset in countries_data.items():
        try:
            v = extractor(dataset)
            if v is not None:
                if isinstance(v, pd.DataFrame):
                    if not v.empty:
                        result.append(country)
                else:
                    result.append(country)
        except Exception:
            pass
    return result


def build_registry_manifest(countries_data: dict) -> list[dict]:
    """Return a JSON-safe manifest of every registry entry with data availability.

    Each entry includes:
      key                – stable identifier (used by LLM + render_charts_by_keys)
      title              – human-readable chart name
      type               – line | bar | scatter | area
      tags               – topic tags for relevance scoring
      source             – data source name
      countries_with_data – list of country names that have data for this chart
    """
    registry = _metric_registry(countries_data)
    manifest: list[dict] = []
    seen_keys: set[str] = set()
    for idx, m in enumerate(registry):
        key = _entry_key(m, idx)
        # Make key unique if collision (shouldn't normally happen)
        if key in seen_keys:
            key = f"{key}_{idx}"
        seen_keys.add(key)
        manifest.append({
            "key": key,
            "title": m.get("title", ""),
            "type": m.get("type", "line"),
            "tags": m.get("tags", []),
            "source": m.get("source", ""),
            "countries_with_data": _countries_with_data(m, countries_data),
            # Keep the original metric dict reference for rendering — not sent to LLM
            "_metric": m,
        })
    return manifest


def render_charts_by_keys(
    keys: list[str],
    manifest: list[dict],
    countries_data: dict,
) -> list[str]:
    """Render exactly the charts identified by `keys`.

    Returns a list of Plotly JSON strings.  Missing / invalid keys produce a
    no-data placeholder so the output always has the same length as `keys`.
    """
    countries = list(countries_data.keys())
    colors = _color_map(countries)
    key_to_metric = {entry["key"]: entry["_metric"] for entry in manifest}
    charts: list[str] = []

    for key in keys:
        metric = key_to_metric.get(key)
        if metric is None:
            fig = _no_data_fig("Chart", f"Chart key '{key}' not found in registry.", "")
            charts.append(fig.to_json())
            continue
        mtype = metric.get("type", "line")
        try:
            if mtype == "line":
                fig = _build_line_chart(
                    countries_data, colors,
                    metric["extractor"],
                    metric["title"], metric["yaxis_title"], metric["source"],
                )
            elif mtype == "bar":
                fig = _build_bar_chart(
                    countries_data, colors,
                    metric["extractor"],
                    metric["title"], metric["yaxis_title"], metric["source"],
                    reference_y=metric.get("reference_y"),
                    reference_label=metric.get("reference_label", ""),
                    ascending=metric.get("ascending", False),
                )
            elif mtype == "scatter":
                fig = _build_scatter_chart(
                    countries_data, colors,
                    metric["x_extractor"], metric["y_extractor"],
                    metric["title"], metric["xaxis_title"], metric["yaxis_title"],
                    metric["source"],
                )
            elif mtype == "area":
                fig = _build_area_chart(
                    countries_data, colors,
                    metric["extractor"],
                    metric["title"], metric["yaxis_title"], metric["source"],
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


# ── Public entry point ─────────────────────────────────────────────────────────

def _indicator_keywords(indicator: str) -> list[str]:
    """Extract searchable keywords from a World Bank indicator name.

    e.g. "internet_users_pct" → ["internet", "users"]
         "life_expectancy"    → ["life", "expectancy"]
         "co2_per_capita"     → ["co2", "capita"]
    """
    # Strip common suffixes that don't help matching
    cleaned = indicator.replace("_pct", "").replace("_usd", "").replace("_gdp", "")
    return [p for p in cleaned.split("_") if len(p) > 2]


def _chart_is_relevant(metric: dict, worldbank_indicators: list[str]) -> bool:
    """Return True if a registry chart entry matches any of the query's WB indicators."""
    if not worldbank_indicators:
        return False
    title_lower = metric.get("title", "").lower()
    data_key    = metric.get("data_key", "")
    for wi in worldbank_indicators:
        # Exact data_key match (e.g. data_key="internet_users" matches "internet_users_pct")
        if data_key and (wi in data_key or data_key in wi):
            return True
        # Keyword match against chart title
        for kw in _indicator_keywords(wi):
            if kw in title_lower:
                return True
    return False


def build_country_comparison_charts(
    countries_data: dict[str, Any],
    selected_tools: list[str],
    query_focus: str,
    worldbank_indicators: list[str] | None = None,
) -> list[str]:
    """
    Build exactly 4 Plotly charts.

    Charts are selected by query relevance first (matching worldbank_indicators),
    then by country coverage, then by visual variety — so the output always shows
    what the user asked about rather than defaulting to GDP/inflation.
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

    # ── Sort candidates: query-relevant first, then by coverage ───────────────
    # Scoring has two components:
    #   1. topic_score  — how many of this chart's tags match the query string
    #   2. tool_bonus   — +1 if the chart's source tool was actually selected,
    #                     0 if the tool wasn't selected (e.g. Open-Meteo / ACLED
    #                     charts when only worldbank was requested)
    # Final sort: (topic_score + tool_bonus DESC, coverage DESC) — so relevant
    # charts from selected tools always beat irrelevant filler from other tools.
    _SOURCE_TOOL_MAP = {
        "world bank":  "worldbank",
        "world bank wgi": "worldbank",
        "ilo":         "employment",
        "ilo / world Bank": "employment",
        "openaq":      "environment",
        "open-meteo":  "environment",
        "acled":       "acled",
        "teleport":    "teleport",
        "unhcr":       "unhcr",
    }

    def _tool_bonus(metric: dict) -> int:
        src = metric.get("source", "").lower()
        for key, tool in _SOURCE_TOOL_MAP.items():
            if key in src:
                return 1 if tool in selected_tools else 0
        return 1  # unknown source — don't penalise

    query_lower = (query_focus or "").lower()
    candidates.sort(
        key=lambda item: (
            _topic_score(item[0], query_lower) + _tool_bonus(item[0]),
            item[1],
        ),
        reverse=True,
    )

    # Three-pass selection:
    #   Pass 1 (variety): at most 1 of each visual type — max 4 distinct types.
    #   Pass 2 (relaxed): cap each type at 2 to fill remaining slots.
    #   Pass 3 (final fill): any leftover candidates, no type cap.
    seen_titles: set[str] = set()
    seen_data_keys: set[str] = set()
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
            # Deduplicate by underlying data — prevents two charts of the same series
            data_key = metric.get("data_key")
            if data_key and data_key in seen_data_keys:
                continue
            mtype = metric.get("type", "line")
            if max_per_type is not None and type_counts.get(mtype, 0) >= max_per_type:
                continue
            seen_titles.add(key)
            if data_key:
                seen_data_keys.add(data_key)
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
