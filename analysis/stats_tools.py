"""Similarity, distance, anomaly helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from geopy.distance import geodesic
from sklearn.metrics.pairwise import cosine_similarity


def run_cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    a = np.asarray(vec_a, dtype=float).reshape(1, -1)
    b = np.asarray(vec_b, dtype=float).reshape(1, -1)
    if a.shape[1] != b.shape[1]:
        n = min(a.shape[1], b.shape[1])
        a, b = a[:, :n], b[:, :n]
    sim = cosine_similarity(a, b)[0, 0]
    return float(sim) if not np.isnan(sim) else 0.0


def haversine_distance(coord_a: tuple[float, float], coord_b: tuple[float, float]) -> float:
    """Great-circle distance in km."""
    return float(geodesic(coord_a, coord_b).kilometers)


def run_anomaly_detect(series: list[float], years: list[int]) -> dict[str, Any]:
    """Z-score |z|>2 flagged as anomalies."""
    s = np.asarray(series, dtype=float)
    y = np.asarray(years, dtype=int)
    if len(s) < 5:
        return {"anomaly_years": [], "z_scores": []}
    mu, sigma = float(np.nanmean(s)), float(np.nanstd(s))
    if sigma == 0:
        return {"anomaly_years": [], "z_scores": []}
    z = (s - mu) / sigma
    anom = [int(yr) for yr, zz in zip(y, z) if abs(zz) > 2]
    return {"anomaly_years": anom, "z_scores": [float(x) for x in z]}
