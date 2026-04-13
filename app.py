"""Flask frontend for the Migration Intelligence Agent."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from flask import Flask, render_template, request
from plotly.io import from_json

from agents.orchestrator import run_migration_pipeline
from analysis.visualization import build_migration_panels
from models.schemas import HypothesisReport

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

app = Flask(__name__)


def _auth_ready() -> bool:
    has_vertex_ai = (
        os.environ.get("GOOGLE_GENAI_USE_VERTEXAI") == "TRUE"
        and os.environ.get("GOOGLE_CLOUD_PROJECT")
    )
    has_api_key = bool(os.environ.get("GOOGLE_API_KEY"))
    return bool(has_vertex_ai or has_api_key or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))


def _normalize_chart(fig_json: str) -> dict[str, Any] | None:
    try:
        fig = from_json(fig_json)
    except Exception:
        return None
    return fig.to_plotly_json()


def _build_context(query: str | None = None) -> dict[str, Any]:
    return {
        "query": query or "",
        "error": None,
        "report": None,
        "overview_chart": None,
        "detail_charts": [],
        "collection_summary": [],
        "push_factor_result": {},
        "destination_result": {},
        "pattern_result": {},
        "relocation_result": {},
        "step_cards": [
            {
                "eyebrow": "Step 1",
                "title": "Collect",
                "body": (
                    "The scout agent retrieves live migration, economic, climate, "
                    "employment, news, AQI, and destination signals from the plan-defined "
                    "sources and writes them into DuckDB-backed state."
                ),
            },
            {
                "eyebrow": "Step 2",
                "title": "Explore And Analyze",
                "body": (
                    "Parallel EDA agents compute correlations, growth shifts, destination "
                    "rankings, lag patterns, and relocation scoring using explicit tool-backed "
                    "calculations over the collected dataset."
                ),
            },
            {
                "eyebrow": "Step 3",
                "title": "Hypothesize",
                "body": (
                    "The hypothesis agent fuses the EDA outputs into a grounded report "
                    "with evidence, competing explanations, confidence, and chart captions."
                ),
            },
        ],
    }


@app.get("/")
def index() -> str:
    return render_template("index.html", **_build_context())


@app.post("/analyze")
def analyze() -> tuple[str, int] | str:
    query = (request.form.get("query") or "").strip()
    context = _build_context(query)

    if not query:
        context["error"] = "Enter a migration question before running the pipeline."
        return render_template("index.html", **context), 400

    if not _auth_ready():
        context["error"] = (
            "Authentication is not configured. Set GOOGLE_API_KEY or "
            "GOOGLE_APPLICATION_CREDENTIALS in .env before running the pipeline."
        )
        return render_template("index.html", **context), 400

    try:
        state = asyncio.run(run_migration_pipeline(query, year_from=None, year_to=None))
    except Exception as exc:
        context["error"] = f"Pipeline execution failed: {exc}"
        return render_template("index.html", **context), 500

    report_raw = state.get("hypothesis_report")
    if not report_raw:
        context["error"] = "The pipeline completed without a hypothesis report."
        return render_template("index.html", **context), 500

    try:
        report = HypothesisReport.model_validate(report_raw)
    except Exception as exc:
        context["error"] = f"Hypothesis report validation failed: {exc}"
        return render_template("index.html", **context), 500

    overview_chart = None
    dataset = state.get("migration_dataset")
    if isinstance(dataset, dict):
        try:
            overview_chart = _normalize_chart(build_migration_panels(dataset))
        except Exception:
            overview_chart = None

    detail_charts = []
    for panel in report.charts:
        normalized = _normalize_chart(panel.fig_json)
        if normalized:
            detail_charts.append(
                {
                    "figure": normalized,
                    "caption": panel.caption,
                    "sources": panel.data_sources,
                }
            )

    context.update(
        {
            "report": report,
            "overview_chart": overview_chart,
            "detail_charts": detail_charts,
            "collection_summary": [
                {"label": "World Bank", "value": len((dataset or {}).get("worldbank", [])) if isinstance(dataset, dict) else 0},
                {"label": "UNHCR", "value": len((dataset or {}).get("displacement", [])) if isinstance(dataset, dict) else 0},
                {"label": "OpenAQ", "value": len((dataset or {}).get("aqi", [])) if isinstance(dataset, dict) else 0},
                {"label": "Open-Meteo", "value": len((dataset or {}).get("climate", [])) if isinstance(dataset, dict) else 0},
                {"label": "ILO / labor", "value": len((dataset or {}).get("employment", [])) if isinstance(dataset, dict) else 0},
                {"label": "News + GDELT", "value": len((dataset or {}).get("news", [])) if isinstance(dataset, dict) else 0},
            ],
            "push_factor_result": state.get("push_factor_result") or {},
            "destination_result": state.get("destination_result") or {},
            "pattern_result": state.get("pattern_result") or {},
            "relocation_result": state.get("relocation_result") or {},
        }
    )
    return render_template("index.html", **context)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
