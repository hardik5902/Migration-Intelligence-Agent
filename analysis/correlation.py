"""EDA statistics: correlation, Granger, lag, growth."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests


def run_correlation_analysis(
    data: dict[str, list[Any]],
    target_col: str,
) -> list[dict[str, Any]]:
    """Pearson and Spearman for each numeric column vs target_col."""
    df = pd.DataFrame(data)
    if target_col not in df.columns:
        return []
    y = pd.to_numeric(df[target_col], errors="coerce")
    out: list[dict[str, Any]] = []
    for col in df.columns:
        if col == target_col:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() < 4:
            continue
        pr = stats.pearsonr(x[mask], y[mask])
        sr = stats.spearmanr(x[mask], y[mask])
        out.append(
            {
                "indicator": col,
                "pearson_r": float(pr.statistic),
                "pearson_p": float(pr.pvalue),
                "spearman_r": float(sr.statistic),
                "spearman_p": float(sr.pvalue),
            }
        )
    out.sort(key=lambda r: abs(r["pearson_r"]), reverse=True)
    return out


def run_granger_test(
    cause: list[float],
    effect: list[float],
    max_lag: int = 4,
) -> dict[str, Any]:
    """Granger causality: does `cause` help predict `effect`."""
    n = min(len(cause), len(effect))
    if n < max_lag + 5:
        return {"lag_years": None, "p_value": None, "is_causal": False, "note": "insufficient_obs"}
    d = pd.DataFrame({"effect": effect[-n:], "cause": cause[-n:]})
    try:
        res = grangercausalitytests(d[["effect", "cause"]], maxlag=max_lag, verbose=False)
        best_lag = None
        best_p = 1.0
        for lag, tests in res.items():
            p = float(tests[0]["ssr_ftest"][1])  # statsmodels tuple: (F, p, ...)
            if p < best_p:
                best_p = p
                best_lag = int(lag)
        return {
            "lag_years": best_lag,
            "p_value": best_p,
            "is_causal": best_p < 0.05,
        }
    except Exception as exc:
        return {"lag_years": None, "p_value": None, "is_causal": False, "note": str(exc)}


def run_lag_analysis(
    series_a: list[float],
    series_b: list[float],
    max_lag: int = 5,
) -> dict[str, Any]:
    """Cross-correlation at lags 1..max_lag (numpy roll)."""
    a = np.asarray(series_a, dtype=float)
    b = np.asarray(series_b, dtype=float)
    n = min(len(a), len(b))
    if n < max_lag + 3:
        return {"optimal_lag": None, "correlation": None}
    a, b = a[-n:], b[-n:]
    best_lag = 0
    best_r = -1.0
    for lag in range(1, max_lag + 1):
        if lag >= n:
            break
        aa = a[:-lag]
        bb = b[lag:]
        if len(aa) < 3:
            continue
        r = np.corrcoef(aa, bb)[0, 1]
        if not np.isnan(r) and abs(r) > abs(best_r):
            best_r = float(r)
            best_lag = lag
    return {"optimal_lag": best_lag, "correlation": best_r}


def run_growth_rate(series: list[float], years: list[int]) -> dict[str, Any]:
    """CAGR over full window + peak YoY."""
    s = np.asarray(series, dtype=float)
    y = np.asarray(years, dtype=int)
    if len(s) < 2:
        return {"cagr": None, "peak_growth_year": None, "trough_year": None}
    order = np.argsort(y)
    s, y = s[order], y[order]
    g = np.diff(s) / np.where(s[:-1] == 0, np.nan, s[:-1])
    peak_idx = int(np.nanargmax(g)) + 1 if np.any(~np.isnan(g)) else None
    trough_idx = int(np.nanargmin(g)) + 1 if np.any(~np.isnan(g)) else None
    years_total = max(int(y[-1] - y[0]), 1)
    # CAGR requires same-sign start/end; negative base ** fraction = complex in Python
    start, end = s[0], s[-1]
    if start == 0 or np.isnan(start) or start * end <= 0:
        cagr = None
    else:
        ratio = float(end / start)
        cagr = (ratio ** (1.0 / years_total)) - 1.0
    return {
        "cagr": float(cagr) if cagr is not None and not np.isnan(cagr) else None,
        "peak_growth_year": int(y[peak_idx]) if peak_idx is not None else None,
        "trough_year": int(y[trough_idx]) if trough_idx is not None else None,
    }
