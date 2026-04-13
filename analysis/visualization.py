"""Plotly charts for the migration UI: intent-aware panels, only when data exists."""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Classified intent → preferred panel order (subset may render if data exists).
INTENT_PANEL_ORDER: dict[str, list[str]] = {
    "push_factor": [
        "macro_displacement",
        "climate",
        "employment",
        "gdelt",
    ],
    "destination": [
        "destinations",
        "displacement_timeline",
        "macro_displacement",
        "climate",
    ],
    "historical": [
        "macro_displacement",
        "climate",
        "gdelt",
        "employment",
    ],
    "real_time": [
        "conflict",
        "news_volume",
        "gdelt",
        "climate",
    ],
    "relocation_advisory": [
        "aqi",
        "city_scores",
        "climate",
        "macro_relocation",
    ],
}

RELOCATION_WB_LABELS = frozenset(
    {
        "health_expenditure_gdp",
        "physicians_per_1000",
        "education_spend_gdp",
        "gdp_per_capita_usd",
        "poverty_headcount",
    }
)

PANEL_TITLES: dict[str, str] = {
    "macro_displacement": "Macro: inflation vs displacement outflow",
    "climate": "Climate: precipitation and temperature anomaly",
    "employment": "Labor market (unemployment)",
    "gdelt": "GDELT media tone over time",
    "destinations": "Destination mix (UNHCR)",
    "displacement_timeline": "Displacement outflow over time",
    "conflict": "Conflict events (ACLED)",
    "news_volume": "News volume over time",
    "aqi": "Air quality (PM2.5 proxy)",
    "city_scores": "Quality-of-life scores (proxies)",
    "macro_relocation": "Relocation macro bundle (World Bank)",
}


def _merge_query_hints(intent: str, user_query: str | None, base: list[str]) -> list[str]:
    """Promote panels implied by the user's wording (does not remove intent defaults)."""
    if not user_query:
        return list(base)
    q = user_query.lower()
    extra: list[str] = []
    if any(w in q for w in ("unemploy", "employment", "job", "labor", "work", "wage")):
        extra.append("employment")
    if any(w in q for w in ("air", "pollut", "aqi", "pm2", "smog", "clean air")):
        extra.append("aqi")
    if any(w in q for w in ("destination", "where are", "going to", "fleeing to", "resettle")):
        extra.append("destinations")
    if any(w in q for w in ("conflict", "war", "violence", "attack", "fighting")):
        extra.append("conflict")
    if any(w in q for w in ("climate", "rain", "drought", "flood", "heat", "temperature")):
        extra.append("climate")
    seen = set()
    out: list[str] = []
    for pid in list(base) + extra:
        if pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
    return out


def _has_macro_displacement(wb: pd.DataFrame, disp: pd.DataFrame) -> bool:
    if not wb.empty and {"year", "label", "value"}.issubset(wb.columns):
        infl = wb[wb["label"] == "inflation"]
        if not infl.empty:
            return True
    if not disp.empty and {"year", "value"}.issubset(disp.columns):
        return not disp.groupby("year", as_index=False)["value"].sum().empty
    return False


def _has_displacement_timeline(disp: pd.DataFrame) -> bool:
    return not disp.empty and {"year", "value"}.issubset(disp.columns)


def _has_climate(cl: pd.DataFrame) -> bool:
    if cl.empty or "year" not in cl.columns:
        return False
    return ("annual_precipitation_mm" in cl.columns) or ("avg_temp_anomaly_c" in cl.columns)


def _has_gdelt(dataset: dict[str, Any]) -> bool:
    gd = pd.DataFrame((dataset.get("gdelt") or {}).get("timeline") or [])
    return not gd.empty and "date" in gd.columns


def _has_employment(em: pd.DataFrame) -> bool:
    return not em.empty and "year" in em.columns and "unemployment_rate" in em.columns


def _has_destinations(dataset: dict[str, Any]) -> bool:
    dest = dataset.get("destinations") or []
    return bool(dest)


def _has_conflict(conf: pd.DataFrame) -> bool:
    return not conf.empty and "event_date" in conf.columns


def _has_news_volume(news: pd.DataFrame) -> bool:
    if news.empty:
        return False
    return "published_at" in news.columns or "title" in news.columns


def _has_aqi(aqi: pd.DataFrame) -> bool:
    if aqi.empty:
        return False
    return "pm25" in aqi.columns or "location" in aqi.columns


def _has_city_scores(cs: pd.DataFrame) -> bool:
    return not cs.empty and "category" in cs.columns and "score_out_of_10" in cs.columns


def _has_macro_relocation(wb: pd.DataFrame) -> bool:
    if wb.empty or "label" not in wb.columns:
        return False
    sub = wb[wb["label"].isin(RELOCATION_WB_LABELS)]
    return not sub.empty


def _check_panel(
    panel_id: str,
    dataset: dict[str, Any],
    wb_primary: pd.DataFrame,
    cl: pd.DataFrame,
    em: pd.DataFrame,
    disp: pd.DataFrame,
    conf: pd.DataFrame,
    news: pd.DataFrame,
    aqi: pd.DataFrame,
    cs: pd.DataFrame,
) -> bool:
    checks: dict[str, Callable[[], bool]] = {
        "macro_displacement": lambda: _has_macro_displacement(wb_primary, disp),
        "displacement_timeline": lambda: _has_displacement_timeline(disp),
        "climate": lambda: _has_climate(cl),
        "employment": lambda: _has_employment(em),
        "gdelt": lambda: _has_gdelt(dataset),
        "destinations": lambda: _has_destinations(dataset),
        "conflict": lambda: _has_conflict(conf),
        "news_volume": lambda: _has_news_volume(news),
        "aqi": lambda: _has_aqi(aqi),
        "city_scores": lambda: _has_city_scores(cs),
        "macro_relocation": lambda: _has_macro_relocation(wb_primary),
    }
    fn = checks.get(panel_id)
    return fn() if fn else False


def _add_macro_displacement(fig: go.Figure, row: int, wb: pd.DataFrame, disp: pd.DataFrame) -> None:
    if not wb.empty and {"year", "label", "value"}.issubset(wb.columns):
        infl = wb[wb["label"] == "inflation"].sort_values("year")
        if not infl.empty:
            fig.add_trace(
                go.Scatter(
                    x=infl["year"],
                    y=infl["value"],
                    name="Inflation %",
                    line={"color": "#b04a35"},
                ),
                row=row,
                col=1,
            )
    if not disp.empty and {"year", "value"}.issubset(disp.columns):
        yearly = disp.groupby("year", as_index=False)["value"].sum().sort_values("year")
        if not yearly.empty:
            fig.add_trace(
                go.Bar(
                    x=yearly["year"],
                    y=yearly["value"],
                    name="Displacement outflow",
                    marker={"color": "#54707b"},
                    opacity=0.55,
                ),
                row=row,
                col=1,
            )


def _add_displacement_timeline(fig: go.Figure, row: int, disp: pd.DataFrame) -> None:
    yearly = disp.groupby("year", as_index=False)["value"].sum().sort_values("year")
    fig.add_trace(
        go.Scatter(
            x=yearly["year"],
            y=yearly["value"],
            name="Outflow",
            line={"color": "#54707b"},
        ),
        row=row,
        col=1,
    )


def _add_climate(fig: go.Figure, row: int, cl: pd.DataFrame) -> None:
    climate = cl.sort_values("year")
    if "annual_precipitation_mm" in climate.columns:
        fig.add_trace(
            go.Bar(
                x=climate["year"],
                y=climate["annual_precipitation_mm"],
                name="Precip mm",
                marker={"color": "#6a78f0"},
            ),
            row=row,
            col=1,
        )
    if "avg_temp_anomaly_c" in climate.columns:
        fig.add_trace(
            go.Scatter(
                x=climate["year"],
                y=climate["avg_temp_anomaly_c"],
                name="Temp anomaly °C",
                line={"color": "#e06c4a"},
            ),
            row=row,
            col=1,
        )


def _add_gdelt(fig: go.Figure, row: int, dataset: dict[str, Any]) -> None:
    gd = pd.DataFrame((dataset.get("gdelt") or {}).get("timeline") or [])
    gd = gd.copy()
    gd["date"] = pd.to_datetime(gd["date"], format="%Y%m%d", errors="coerce")
    gd = gd.dropna(subset=["date"])
    if "tone" in gd.columns and not gd.empty:
        tone = gd.groupby("date", as_index=False)["tone"].mean()
        fig.add_trace(
            go.Scatter(
                x=tone["date"],
                y=tone["tone"],
                name="GDELT tone",
                line={"color": "#2c8a7e"},
            ),
            row=row,
            col=1,
        )


def _add_employment(fig: go.Figure, row: int, em: pd.DataFrame) -> None:
    em = em.sort_values("year")
    if "unemployment_rate" in em.columns:
        fig.add_trace(
            go.Scatter(
                x=em["year"],
                y=em["unemployment_rate"],
                name="Unemployment",
                line={"color": "#0f766e"},
            ),
            row=row,
            col=1,
        )
    if "youth_unemployment_rate" in em.columns and em["youth_unemployment_rate"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=em["year"],
                y=em["youth_unemployment_rate"],
                name="Youth unemployment",
                line={"color": "#cc5f1a"},
            ),
            row=row,
            col=1,
        )


def _add_destinations(fig: go.Figure, row: int, dataset: dict[str, Any]) -> None:
    rows = dataset.get("destinations") or []
    if not rows:
        return
    df = pd.DataFrame(rows)
    name_col = "destination_country" if "destination_country" in df.columns else df.columns[0]
    val_col = "refugee_count" if "refugee_count" in df.columns else None
    if val_col is None:
        return
    sub = df[[name_col, val_col]].dropna().sort_values(val_col, ascending=False).head(20)
    fig.add_trace(
        go.Bar(
            x=sub[name_col],
            y=sub[val_col],
            name="Refugees",
            marker={"color": "#5b4f9a"},
        ),
        row=row,
        col=1,
    )


def _add_conflict(fig: go.Figure, row: int, conf: pd.DataFrame) -> None:
    c = conf.copy()
    c["event_date"] = pd.to_datetime(c["event_date"], errors="coerce")
    c = c.dropna(subset=["event_date"])
    if c.empty:
        return
    c["month"] = c["event_date"].dt.to_period("M").dt.to_timestamp()
    monthly = c.groupby("month", as_index=False).size().rename(columns={"size": "events"})
    fig.add_trace(
        go.Bar(x=monthly["month"], y=monthly["events"], name="Events / month", marker={"color": "#8b2942"}),
        row=row,
        col=1,
    )


def _add_news_volume(fig: go.Figure, row: int, news: pd.DataFrame) -> None:
    n = news.copy()
    if "published_at" in n.columns:
        n["published_at"] = pd.to_datetime(n["published_at"], errors="coerce")
        n = n.dropna(subset=["published_at"])
        n["year"] = n["published_at"].dt.year
    else:
        n["year"] = 0
    if n.empty:
        return
    counts = n.groupby("year", as_index=False).size().rename(columns={"size": "articles"})
    counts = counts[counts["year"] > 0]
    if counts.empty:
        return
    fig.add_trace(
        go.Bar(x=counts["year"], y=counts["articles"], name="Articles", marker={"color": "#3d6b8a"}),
        row=row,
        col=1,
    )


def _add_aqi(fig: go.Figure, row: int, aqi: pd.DataFrame) -> None:
    df = aqi.copy()
    if "pm25" not in df.columns:
        return
    if "location" in df.columns:
        label = df["location"]
        if "city" in df.columns:
            label = label.fillna(df["city"])
    elif "city" in df.columns:
        label = df["city"]
    else:
        label = pd.Series(range(len(df))).astype(str)
    fig.add_trace(
        go.Bar(
            x=label.astype(str)[:40],
            y=df["pm25"][:40],
            name="PM2.5",
            marker={"color": "#c45c26"},
        ),
        row=row,
        col=1,
    )


def _add_city_scores(fig: go.Figure, row: int, cs: pd.DataFrame) -> None:
    df = cs.groupby("category", as_index=False)["score_out_of_10"].mean().sort_values("score_out_of_10", ascending=False)
    fig.add_trace(
        go.Bar(
            x=df["category"],
            y=df["score_out_of_10"],
            name="Score /10",
            marker={"color": "#2d6a4f"},
        ),
        row=row,
        col=1,
    )


def _add_macro_relocation(fig: go.Figure, row: int, wb: pd.DataFrame) -> None:
    sub = wb[wb["label"].isin(RELOCATION_WB_LABELS)].sort_values(["label", "year"])
    for label, grp in sub.groupby("label"):
        fig.add_trace(
            go.Scatter(
                x=grp["year"],
                y=grp["value"],
                name=str(label)[:28],
                line={"width": 2},
            ),
            row=row,
            col=1,
        )


def _dispatch(
    panel_id: str,
    fig: go.Figure,
    row: int,
    dataset: dict[str, Any],
    wb_primary: pd.DataFrame,
    cl: pd.DataFrame,
    em: pd.DataFrame,
    disp: pd.DataFrame,
    conf: pd.DataFrame,
    news: pd.DataFrame,
    aqi: pd.DataFrame,
    cs: pd.DataFrame,
) -> None:
    if panel_id == "macro_displacement":
        _add_macro_displacement(fig, row, wb_primary, disp)
    elif panel_id == "displacement_timeline":
        _add_displacement_timeline(fig, row, disp)
    elif panel_id == "climate":
        _add_climate(fig, row, cl)
    elif panel_id == "employment":
        _add_employment(fig, row, em)
    elif panel_id == "gdelt":
        _add_gdelt(fig, row, dataset)
    elif panel_id == "destinations":
        _add_destinations(fig, row, dataset)
    elif panel_id == "conflict":
        _add_conflict(fig, row, conf)
    elif panel_id == "news_volume":
        _add_news_volume(fig, row, news)
    elif panel_id == "aqi":
        _add_aqi(fig, row, aqi)
    elif panel_id == "city_scores":
        _add_city_scores(fig, row, cs)
    elif panel_id == "macro_relocation":
        _add_macro_relocation(fig, row, wb_primary)


def build_migration_panels(
    dataset: dict[str, Any],
    *,
    intent: str | None = None,
    user_query: str | None = None,
) -> str:
    """
    Build a variable-height figure with one row per panel that (1) matches intent
    (and optional query hints) and (2) has data. No empty 2×2 grid.
    """
    intent_key = (intent or dataset.get("intent") or "push_factor").strip().lower()
    if intent_key not in INTENT_PANEL_ORDER:
        intent_key = "push_factor"

    base_order = INTENT_PANEL_ORDER[intent_key]
    order = _merge_query_hints(intent_key, user_query, base_order)

    wb = pd.DataFrame(dataset.get("worldbank") or [])
    cc = (dataset.get("country_code") or "").strip()
    if cc and not wb.empty and "country" in wb.columns:
        wb_primary = wb[wb["country"].astype(str).str.upper() == cc.upper()].copy()
        if wb_primary.empty:
            wb_primary = wb.copy()
    else:
        wb_primary = wb.copy()
    cl = pd.DataFrame(dataset.get("climate") or [])
    em = pd.DataFrame(dataset.get("employment") or [])
    disp = pd.DataFrame(dataset.get("displacement") or [])
    conf = pd.DataFrame(dataset.get("conflict_events") or [])
    news = pd.DataFrame(dataset.get("news") or [])
    aqi = pd.DataFrame(dataset.get("aqi") or [])
    cs = pd.DataFrame(dataset.get("city_scores") or [])

    chosen: list[str] = []
    for pid in order:
        if _check_panel(pid, dataset, wb_primary, cl, em, disp, conf, news, aqi, cs):
            chosen.append(pid)

    country = dataset.get("country") or dataset.get("country_code") or "Country"
    title = f"Exploratory charts · {country} · intent={intent_key}"

    if not chosen:
        fig = go.Figure()
        fig.add_annotation(
            text="No chartable series for this query and country (sources returned no rows for the selected panels).",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 14},
        )
        fig.update_layout(
            height=320,
            title_text=title,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#fffdf8",
            margin={"l": 32, "r": 20, "t": 56, "b": 28},
        )
        return fig.to_json()

    n = len(chosen)
    subtitles = tuple(PANEL_TITLES.get(p, p) for p in chosen)
    fig = make_subplots(
        rows=n,
        cols=1,
        subplot_titles=subtitles,
        vertical_spacing=min(0.12, 0.04 + 0.02 * n),
    )

    for i, pid in enumerate(chosen, start=1):
        _dispatch(pid, fig, i, dataset, wb_primary, cl, em, disp, conf, news, aqi, cs)

    height = min(200 + 200 * n, 1400)
    fig.update_layout(
        height=height,
        title_text=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#fffdf8",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 32, "r": 20, "t": 72, "b": 28},
    )
    return fig.to_json()
