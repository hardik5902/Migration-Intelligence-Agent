"""EDA-specific Plotly chart builders.

Four distinct chart types — each visually different from the comparison charts
in country_charts.py to ensure variety:

  heatmap         → metric-metric Pearson correlation heatmap (go.Heatmap)
  growth_rate_bar → horizontal CAGR bar with diverging green/red colours
  anomaly_timeline→ line chart with star markers on anomalous years
  distribution_box→ bar chart with error bars (mean ± 1 σ per country)

build_eda_charts() selects up to 2 charts driven by the charts_hint list
produced by run_eda(), so different queries produce different EDA visuals.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

COUNTRY_COLORS = [
    "#af3f2f", "#54707b", "#6a78f0", "#2d6a4f",
    "#e06c4a", "#5b4f9a", "#3d6b8a", "#c45c26",
]


# ---------------------------------------------------------------------------
# Shared layout factory
# ---------------------------------------------------------------------------

def _base_layout(title: str, height: int = 460, source: str = "") -> dict:
    annotations = []
    if source:
        annotations.append(
            dict(
                text=f"Source: {source}",
                xref="paper", yref="paper",
                x=1, y=-0.38,
                xanchor="right", yanchor="top",
                showarrow=False,
                font=dict(size=9, color="#9b8f84"),
            )
        )
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(250,248,244,0.95)",
        height=height,
        autosize=True,
        margin=dict(l=80, r=36, t=64, b=130),
        font=dict(family="Space Grotesk, sans-serif", size=12, color="#1e1b18"),
        title=dict(
            text=title,
            font=dict(size=13, color="#1e1b18", family="Source Serif 4, serif"),
            x=0,
            xanchor="left",
            pad=dict(l=4, b=6),
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
        annotations=annotations,
    )


# ---------------------------------------------------------------------------
# Chart 1 — Correlation Heatmap  (go.Heatmap)
# ---------------------------------------------------------------------------

def _build_correlation_heatmap(latest_matrix: dict[str, dict]) -> go.Figure | None:
    """Pearson correlation matrix across all available indicators (latest values)."""
    if not latest_matrix:
        return None

    # latest_matrix is {metric: {country: value}} — transpose to {country: {metric: value}}
    df = pd.DataFrame(latest_matrix)        # rows = countries, cols = metrics
    numeric = df.select_dtypes(include=[float, int]).dropna(axis=1, how="all")

    if numeric.shape[1] < 2 or numeric.shape[0] < 3:
        return None

    # Use complete rows for correlation
    complete = numeric.dropna()
    if len(complete) < 3:
        return None

    corr = complete.corr(method="pearson")
    if corr.empty:
        return None

    labels = [lbl.replace("_", " ").title() for lbl in corr.columns]
    z = corr.values.tolist()

    fig = go.Figure(go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        colorscale="RdBu_r",
        zmid=0,
        zmin=-1,
        zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in z],
        texttemplate="%{text}",
        textfont={"size": 11, "color": "rgba(30,27,24,0.85)"},
        hovertemplate="<b>%{y} × %{x}</b><br>Pearson r = %{z:.3f}<extra></extra>",
        showscale=True,
        colorbar=dict(
            title=dict(text="Pearson r", side="right"),
            thickness=14,
            len=0.85,
            tickfont=dict(size=10),
        ),
    ))

    n_metrics = len(labels)
    cell_px = max(52, min(72, 400 // max(n_metrics, 1)))
    h = max(460, cell_px * n_metrics + 160)

    layout = _base_layout(
        "Indicator correlation matrix — latest values across countries",
        height=h,
        source="World Bank · ILO · Open-Meteo (EDA)",
    )
    layout.pop("legend", None)
    layout["margin"] = dict(l=140, r=80, t=64, b=140)
    layout["xaxis"] = dict(side="bottom", tickangle=-40, automargin=True, tickfont=dict(size=11))
    layout["yaxis"] = dict(automargin=True, tickfont=dict(size=11))
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Chart 2 — CAGR Growth Rate Horizontal Bar  (go.Bar, orientation="h")
# ---------------------------------------------------------------------------

def _build_cagr_bar(
    growth_rates: dict[str, dict],
    metric: str = "gdp_growth",
) -> go.Figure | None:
    """Horizontal diverging bar: positive growth = green, negative = red."""
    cagrs: dict[str, float] = {}
    for country, grs in growth_rates.items():
        gr = grs.get(metric, {})
        cagr = gr.get("cagr")
        if cagr is not None and not np.isnan(cagr):
            cagrs[country] = cagr * 100.0

    if len(cagrs) < 2:
        return None

    # Sort ascending so highest appears at top of horizontal bar
    sorted_items = sorted(cagrs.items(), key=lambda x: x[1])
    countries = [k for k, _ in sorted_items]
    values   = [v for _, v in sorted_items]
    colors   = ["#2d6a4f" if v >= 0 else "#af3f2f" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=countries,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.1f}%" for v in values],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>CAGR: %{x:+.2f}%<extra></extra>",
        showlegend=False,
    ))

    fig.add_vline(
        x=0,
        line_dash="dot",
        line_color="rgba(30,27,24,0.25)",
        line_width=1.5,
    )

    metric_label = metric.replace("_", " ").title()
    h = max(460, 100 + 58 * len(countries))
    layout = _base_layout(
        f"Growth trajectory: {metric_label} — CAGR comparison across countries",
        height=h,
        source="World Bank / ILO (EDA)",
    )
    layout.pop("legend", None)
    layout["margin"] = dict(l=140, r=100, t=64, b=120)
    layout["xaxis"] = dict(
        title="Compound Annual Growth Rate (%)",
        automargin=True,
        gridcolor="rgba(30,27,24,0.07)",
        zerolinecolor="rgba(30,27,24,0.3)",
        zerolinewidth=1.5,
        ticksuffix="%",
        tickfont=dict(size=11),
    )
    layout["yaxis"] = dict(automargin=True, tickfont=dict(size=11))
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Chart 3 — Anomaly-Annotated Time Series  (go.Scatter + star markers)
# ---------------------------------------------------------------------------

def _wb_series_local(dataset: dict, label: str) -> tuple[list[float], list[int]] | None:
    import pandas as _pd
    wb = _pd.DataFrame(dataset.get("worldbank") or [])
    if wb.empty or "label" not in wb.columns:
        return None
    sub = wb[wb["label"] == label].dropna(subset=["value", "year"]).sort_values("year")
    if sub.empty:
        return None
    return sub["value"].astype(float).tolist(), sub["year"].astype(int).tolist()


def _emp_series_local(dataset: dict, col: str) -> tuple[list[float], list[int]] | None:
    import pandas as _pd
    emp = _pd.DataFrame(dataset.get("employment") or [])
    if emp.empty or col not in emp.columns:
        return None
    sub = emp.dropna(subset=[col, "year"]).sort_values("year")
    if sub.empty:
        return None
    return sub[col].astype(float).tolist(), sub["year"].astype(int).tolist()


def _clim_series_local(dataset: dict, col: str) -> tuple[list[float], list[int]] | None:
    import pandas as _pd
    clim = _pd.DataFrame(dataset.get("climate") or [])
    if clim.empty or col not in clim.columns:
        return None
    sub = clim.dropna(subset=[col, "year"]).sort_values("year")
    if sub.empty:
        return None
    return sub[col].astype(float).tolist(), sub["year"].astype(int).tolist()


def _fetch_series(dataset: dict, metric: str) -> tuple[list[float], list[int]]:
    """Try worldbank label, then employment col, then climate col."""
    for fn in (_wb_series_local, _emp_series_local, _clim_series_local):
        result = fn(dataset, metric)
        if result:
            return result
    return [], []


def _build_anomaly_timeline(
    countries_data: dict[str, Any],
    anomalies: dict[str, list],
    metric_name: str = "gdp_growth",
) -> go.Figure | None:
    """Line chart with ★ markers on years where |z-score| > 2."""
    # Pick a metric that actually has anomalies, falling back to any available
    chosen_metric = metric_name
    has_anomaly_for_metric = any(
        any(a["metric"] == metric_name for a in anom_list)
        for anom_list in anomalies.values()
        if anom_list
    )
    if not has_anomaly_for_metric:
        for _, anom_list in anomalies.items():
            if anom_list:
                chosen_metric = anom_list[0]["metric"]
                break

    fig = go.Figure()
    has_data = False

    for i, (country, dataset) in enumerate(countries_data.items()):
        vals, years = _fetch_series(dataset, chosen_metric)
        if not vals:
            continue

        color = COUNTRY_COLORS[i % len(COUNTRY_COLORS)]

        # Main trend line
        fig.add_trace(go.Scatter(
            x=years,
            y=vals,
            name=country,
            mode="lines+markers",
            line=dict(color=color, width=2.5),
            marker=dict(size=5, symbol="circle"),
            hovertemplate=(
                f"<b>{country}</b><br>Year: %{{x}}<br>"
                f"{chosen_metric.replace('_', ' ').title()}: %{{y:.2f}}<extra></extra>"
            ),
        ))
        has_data = True

        # Anomaly star overlay
        anom_year_set: set[int] = set()
        for a in anomalies.get(country, []):
            if a["metric"] == chosen_metric:
                anom_year_set.update(a["years"])

        if anom_year_set:
            ax = [yr for yr in years if yr in anom_year_set]
            ay = [v  for yr, v in zip(years, vals) if yr in anom_year_set]
            if ax:
                fig.add_trace(go.Scatter(
                    x=ax,
                    y=ay,
                    name=f"{country} outlier",
                    mode="markers",
                    marker=dict(
                        size=16,
                        symbol="star",
                        color=color,
                        line=dict(color="#1e1b18", width=1.5),
                    ),
                    hovertemplate=(
                        f"<b>{country} — OUTLIER</b><br>"
                        "Year: %{x}<br>|z-score| > 2: %{y:.2f}<extra></extra>"
                    ),
                ))

    if not has_data:
        return None

    metric_label = chosen_metric.replace("_", " ").title()
    layout = _base_layout(
        f"Anomaly detection: {metric_label} (★ = statistical outlier, |z| > 2σ)",
        height=480,
        source="EDA — Z-score detection",
    )
    layout["yaxis"] = dict(
        title=metric_label,
        automargin=True,
        gridcolor="rgba(30,27,24,0.07)",
        zerolinecolor="rgba(30,27,24,0.15)",
        tickfont=dict(size=11),
    )
    layout["xaxis"] = dict(
        title="Year",
        automargin=True,
        gridcolor="rgba(30,27,24,0.07)",
        tickfont=dict(size=11),
    )
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Chart 4 — Statistical Spread: Mean ± 1σ Bar  (go.Bar with error_y)
# ---------------------------------------------------------------------------

def _build_stats_spread(
    stats_summary: dict[str, dict],
    metric: str = "gdp_growth",
) -> go.Figure | None:
    """Bar chart showing mean ± 1 standard deviation per country."""
    countries_with_data = [
        c for c in stats_summary
        if stats_summary[c].get(metric) and stats_summary[c][metric].get("n", 0) >= 2
    ]
    if len(countries_with_data) < 2:
        return None

    countries_sorted = sorted(
        countries_with_data,
        key=lambda c: stats_summary[c][metric]["mean"],
        reverse=True,
    )

    means  = [stats_summary[c][metric]["mean"] for c in countries_sorted]
    stds   = [stats_summary[c][metric]["std"]  for c in countries_sorted]
    colors = [COUNTRY_COLORS[i % len(COUNTRY_COLORS)] for i in range(len(countries_sorted))]

    fig = go.Figure(go.Bar(
        x=countries_sorted,
        y=means,
        error_y=dict(
            type="data",
            array=stds,
            visible=True,
            color="rgba(30,27,24,0.45)",
            thickness=2,
            width=6,
        ),
        marker_color=colors,
        text=[f"{m:.1f}" for m in means],
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Mean: %{y:.2f}<br>"
            "σ: %{error_y.array:.2f}<extra></extra>"
        ),
        showlegend=False,
    ))

    metric_label = metric.replace("_", " ").title()
    layout = _base_layout(
        f"Distribution summary: {metric_label} — mean ± 1σ across analysis years",
        height=480,
        source="EDA statistical aggregation",
    )
    layout.pop("legend", None)
    layout["yaxis"] = dict(
        title=metric_label,
        automargin=True,
        gridcolor="rgba(30,27,24,0.07)",
        zerolinecolor="rgba(30,27,24,0.15)",
        tickfont=dict(size=11),
    )
    layout["xaxis"] = dict(automargin=True, tickfont=dict(size=11))
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_eda_charts(
    eda_findings: dict[str, Any],
    countries_data: dict[str, Any],
) -> list[str]:
    """
    Build up to 2 EDA-specific charts driven by eda_findings["charts_hint"].

    Chart types are chosen to be visually distinct from the comparison charts
    (which are plain line / bar charts per country over time).

    Returns a list of Plotly JSON strings.
    """
    growth_rates  = eda_findings.get("growth_rates", {})
    anomalies     = eda_findings.get("anomalies", {})
    stats_summary = eda_findings.get("stats_summary", {})
    latest_matrix = eda_findings.get("latest_matrix", {})
    charts_hint   = eda_findings.get("charts_hint", [])

    charts: list[go.Figure] = []
    attempted: set[str] = set()

    def _try(hint: str) -> go.Figure | None:
        if hint == "correlation_heatmap":
            return _build_correlation_heatmap(latest_matrix)
        if hint == "growth_rate_bar":
            # Priority: conflict/displacement level series first, then economic
            best = next(
                (m for m in [
                    "displacement", "conflict_events", "fatalities",
                    "gdp_per_capita", "unemployment", "temp_anomaly", "gdp_growth",
                ]
                 if any(
                     growth_rates.get(c, {}).get(m, {}).get("cagr") is not None
                     for c in growth_rates
                 )),
                "gdp_per_capita",
            )
            return _build_cagr_bar(growth_rates, best)
        if hint == "anomaly_timeline":
            # Pick anomaly in the most relevant metric (first one found)
            anomaly_metric = next(
                (anom_list[0]["metric"]
                 for anom_list in anomalies.values() if anom_list),
                "gdp_growth",
            )
            return _build_anomaly_timeline(countries_data, anomalies, anomaly_metric)
        if hint == "distribution_box":
            # Priority: conflict/safety metrics first, then economic
            best = next(
                (m for m in [
                    "conflict_events", "fatalities", "political_stability",
                    "displacement", "unemployment", "gdp_growth", "inflation", "temp_anomaly",
                ]
                 if sum(1 for c in stats_summary if stats_summary[c].get(m, {}).get("n", 0) >= 2) >= 2),
                "gdp_growth",
            )
            return _build_stats_spread(stats_summary, best)
        return None

    for hint in charts_hint:
        if len(charts) >= 2:
            break
        if hint in attempted:
            continue
        attempted.add(hint)
        fig = _try(hint)
        if fig is not None:
            charts.append(fig)

    # Fallback: prefer statistical charts (distribution/heatmap) over time-series
    if not charts:
        for fallback in ["distribution_box", "correlation_heatmap", "anomaly_timeline", "growth_rate_bar"]:
            if fallback not in attempted:
                fig = _try(fallback)
                if fig is not None:
                    charts.append(fig)
                    if len(charts) >= 2:
                        break

    return [fig.to_json() for fig in charts]
