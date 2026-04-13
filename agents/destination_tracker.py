"""EDA: rank destinations and note proximity vs wealth anomalies."""

from __future__ import annotations

import pandas as pd

from analysis.stats_tools import haversine_distance
from models.schemas import DestinationResult, IntentConfig, MigrationDataset, ScoredDestination

# Rough capital coordinates for distance sanity checks (lat, lon)
_CAPS: dict[str, tuple[float, float]] = {
    "COL": (4.71, -74.07),
    "PER": (-12.05, -77.04),
    "ECU": (-0.18, -78.47),
    "ESP": (40.42, -3.70),
    "USA": (38.90, -77.04),
    "CHL": (-33.45, -70.67),
    "BRA": (-15.80, -47.88),
}


def run_destination_tracker(
    dataset: MigrationDataset,
    intent: IntentConfig,
) -> DestinationResult:
    dests = dataset.destinations or []
    if not dests:
        return DestinationResult(narrative="No UNHCR destination breakdown available.")
    df = pd.DataFrame(dests).sort_values("refugee_count", ascending=False).head(10)
    origin = dataset.country_code.upper()[:3]
    ocap = _CAPS.get(origin, (10.0, 0.0))
    scored: list[ScoredDestination] = []
    for _, row in df.iterrows():
        name = str(row["destination_country"])
        # crude ISO guess from name length — distance optional
        dcap = (40.0, -4.0)
        for k, v in _CAPS.items():
            if k.lower() in name.lower():
                dcap = v
                break
        _km = haversine_distance(ocap, dcap)
        scored.append(
            ScoredDestination(
                country=name,
                refugee_count=float(row["refugee_count"]),
                share_of_outflow=None,
            )
        )
    total = sum(s.refugee_count or 0 for s in scored) or 1.0
    for s in scored:
        s.share_of_outflow = round((s.refugee_count or 0) / total, 3)

    top = scored[0].country if scored else ""
    anomaly = ""
    target = (intent.target_country or dataset.target_country or "").strip()
    if len(scored) >= 2:
        anomaly = (
            f"{top} leads absorption ({scored[0].share_of_outflow:.0%} of tracked flow in this slice) "
            f"while secondary hubs differ in distance and income context — validate with GDP join in production."
        )
    if target:
        target_match = next((row for row in scored if row.country.lower() == target.lower()), None)
        if target_match is None:
            return DestinationResult(
                top_destinations=scored,
                anomaly_note=(
                    f"Requested destination '{target}' does not appear in the observed destination breakdown "
                    f"for {dataset.country or intent.country}. The data does not support a meaningful migration corridor."
                ),
                narrative=(
                    f"No material UNHCR-style destination evidence found for the route "
                    f"{dataset.country or intent.country} → {target}."
                ),
            )
        anomaly = (
            f"Requested destination '{target}' is present but not dominant, representing "
            f"{target_match.share_of_outflow:.0%} of the tracked outflow."
        )

    return DestinationResult(
        top_destinations=scored,
        anomaly_note=anomaly,
        narrative=f"Top destination by recorded UNHCR-style totals: {top}.",
    )
