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
# Manifest + dispatcher (LLM-driven EDA chart selection)
# ---------------------------------------------------------------------------

_EDA_CHART_META = {
    "correlation_heatmap": {
        "title": "Indicator correlation matrix — latest values across countries",
        "type":  "heatmap",
        "desc":  "Pearson correlation between every pair of available indicators.",
    },
    "growth_rate_bar": {
        "title": "CAGR comparison across countries",
        "type":  "bar",
        "desc":  "Compound annual growth rate for a single metric, sorted by value.",
    },
    "anomaly_timeline": {
        "title": "Anomaly-annotated time series (|z| > 2σ)",
        "type":  "line",
        "desc":  "Trend line with star markers on statistically anomalous years.",
    },
    "distribution_box": {
        "title": "Mean ± 1σ distribution summary",
        "type":  "bar",
        "desc":  "Error bars show statistical spread for a single metric per country.",
    },
}


def _metric_label(metric: str) -> str:
    return metric.replace("_", " ").title()


def _build_dispatcher(eda_findings: dict, countries_data: dict):
    growth_rates  = eda_findings.get("growth_rates", {})
    anomalies     = eda_findings.get("anomalies", {})
    stats_summary = eda_findings.get("stats_summary", {})
    latest_matrix = eda_findings.get("latest_matrix", {})

    def render(key: str) -> go.Figure | None:
        if key == "correlation_heatmap":
            return _build_correlation_heatmap(latest_matrix)
        if ":" not in key:
            return None
        base, metric = key.split(":", 1)
        if base == "growth_rate_bar":
            return _build_cagr_bar(growth_rates, metric)
        if base == "anomaly_timeline":
            return _build_anomaly_timeline(countries_data, anomalies, metric)
        if base == "distribution_box":
            return _build_stats_spread(stats_summary, metric)
        return None

    return render


def build_eda_chart_manifest(
    eda_findings: dict[str, Any],
    countries_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Enumerate every EDA chart that can actually be built from the data.

    Returns a JSON-safe list of entries the unified analysis LLM can choose from:
      { key, title, type, description, metric, applicable_countries }
    """
    growth_rates           = eda_findings.get("growth_rates", {})
    anomalies              = eda_findings.get("anomalies", {})
    stats_summary          = eda_findings.get("stats_summary", {})
    latest_matrix          = eda_findings.get("latest_matrix", {})
    active_metrics_ordered = eda_findings.get("active_metrics_ordered", [])

    entries: list[dict[str, Any]] = []

    # Correlation heatmap: needs ≥3 countries × ≥2 numeric metrics
    if latest_matrix:
        df = pd.DataFrame(latest_matrix)
        numeric = df.select_dtypes(include=[float, int]).dropna(axis=1, how="all")
        if numeric.shape[0] >= 3 and numeric.shape[1] >= 2 and len(numeric.dropna()) >= 3:
            entries.append({
                "key":   "correlation_heatmap",
                "title": _EDA_CHART_META["correlation_heatmap"]["title"],
                "type":  _EDA_CHART_META["correlation_heatmap"]["type"],
                "description": _EDA_CHART_META["correlation_heatmap"]["desc"],
                "metric": None,
                "applicable_countries": list(numeric.dropna().index),
            })

    # Per-metric entries in query-priority order
    for metric in active_metrics_ordered:
        label = _metric_label(metric)

        # distribution_box: ≥2 countries with n ≥ 2
        countries_with_spread = [
            c for c in stats_summary
            if stats_summary.get(c, {}).get(metric, {}).get("n", 0) >= 2
        ]
        if len(countries_with_spread) >= 2:
            entries.append({
                "key":   f"distribution_box:{metric}",
                "title": f"Distribution summary — {label} (mean ± 1σ)",
                "type":  "bar",
                "description": _EDA_CHART_META["distribution_box"]["desc"],
                "metric": metric,
                "applicable_countries": countries_with_spread,
            })

        # growth_rate_bar: ≥2 countries with CAGR value
        countries_with_cagr = [
            c for c in growth_rates
            if growth_rates.get(c, {}).get(metric, {}).get("cagr") is not None
        ]
        if len(countries_with_cagr) >= 2:
            entries.append({
                "key":   f"growth_rate_bar:{metric}",
                "title": f"CAGR comparison — {label}",
                "type":  "bar",
                "description": _EDA_CHART_META["growth_rate_bar"]["desc"],
                "metric": metric,
                "applicable_countries": countries_with_cagr,
            })

        # anomaly_timeline: at least one country with an anomaly for this metric
        countries_with_anom = [
            c for c, lst in anomalies.items()
            if any(a.get("metric") == metric for a in (lst or []))
        ]
        if countries_with_anom:
            entries.append({
                "key":   f"anomaly_timeline:{metric}",
                "title": f"Anomaly timeline — {label}",
                "type":  "line",
                "description": _EDA_CHART_META["anomaly_timeline"]["desc"],
                "metric": metric,
                "applicable_countries": countries_with_anom,
            })

    return entries


def render_eda_charts_by_keys(
    keys: list[str],
    eda_findings: dict[str, Any],
    countries_data: dict[str, Any],
) -> list[str]:
    """Render exactly the EDA charts identified by `keys` (max 4).

    Keys that fail to render are silently skipped — the caller can pad or not.
    Returns a list of Plotly JSON strings, length ≤ len(keys).
    """
    render = _build_dispatcher(eda_findings, countries_data)
    out: list[str] = []
    for key in keys[:4]:
        try:
            fig = render(key)
        except Exception as exc:
            print(f"[EDA_CHARTS] render failed for {key}: {exc}")
            fig = None
        if fig is not None:
            out.append(fig.to_json())
    return out


# ---------------------------------------------------------------------------
# Public entry point (legacy fallback — heuristic, max 4 charts)
# ---------------------------------------------------------------------------

def build_eda_charts(
    eda_findings: dict[str, Any],
    countries_data: dict[str, Any],
    max_charts: int = 4,
) -> list[str]:
    """
    Heuristic fallback: build up to `max_charts` EDA charts from the manifest.

    Used only when the unified analysis LLM cannot select EDA chart keys.
    Prefers the first N entries of the manifest, which is already sorted in
    query-priority order by `build_eda_chart_manifest`.
    """
    manifest = build_eda_chart_manifest(eda_findings, countries_data)
    if not manifest:
        return []

    # Prefer variety across chart types — dedupe by base key
    seen_bases: set[str] = set()
    picked: list[str] = []
    for entry in manifest:
        base = entry["key"].split(":", 1)[0]
        if base in seen_bases and len(picked) >= 2:
            continue
        picked.append(entry["key"])
        seen_bases.add(base)
        if len(picked) >= max_charts:
            break

    return render_eda_charts_by_keys(picked, eda_findings, countries_data)
