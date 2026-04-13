"""Flask frontend for the Migration Intelligence — Country Comparison pipeline."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from flask import Flask, render_template, request
from plotly.io import from_json

from agents.country_pipeline import AVAILABLE_TOOLS, run_country_pipeline

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

app = Flask(__name__)

# Tool metadata shown in the UI
TOOL_CARDS = [
    {
        "id": "worldbank",
        "name": "World Bank",
        "icon": "📊",
        "description": "GDP per capita, inflation, health & education spending, poverty rates",
    },
    {
        "id": "unhcr",
        "name": "UNHCR",
        "icon": "🌍",
        "description": "Refugee displacement outflows and top destination countries",
    },
    {
        "id": "acled",
        "name": "ACLED",
        "icon": "⚔️",
        "description": "Armed conflict events, fatalities, and crisis locations",
    },
    {
        "id": "teleport",
        "name": "Teleport",
        "icon": "🏙️",
        "description": "City quality-of-life: housing, safety, education, healthcare scores",
    },
    {
        "id": "news",
        "name": "News + GDELT",
        "icon": "📰",
        "description": "News coverage volume and media sentiment over time",
    },
    {
        "id": "climate",
        "name": "Open-Meteo",
        "icon": "🌡️",
        "description": "Temperature anomalies and annual precipitation trends",
    },
    {
        "id": "employment",
        "name": "ILO Labor",
        "icon": "💼",
        "description": "Unemployment rates, youth unemployment, labour participation",
    },
    {
        "id": "aqi",
        "name": "OpenAQ",
        "icon": "💨",
        "description": "Air quality PM2.5 measurements by city and location",
    },
]


def _auth_ready() -> bool:
    has_vertex = (
        os.environ.get("GOOGLE_GENAI_USE_VERTEXAI") == "TRUE"
        and os.environ.get("GOOGLE_CLOUD_PROJECT")
    )
    return bool(
        has_vertex
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    )


def _normalize_chart(fig_json: str) -> dict[str, Any] | None:
    try:
        fig = from_json(fig_json)
        return fig.to_plotly_json()
    except Exception:
        return None


def _base_context(query: str = "") -> dict[str, Any]:
    return {
        "query": query,
        "error": None,
        "result": None,
        "charts": [],
        "tool_cards": TOOL_CARDS,
    }


@app.get("/")
def index() -> str:
    return render_template("index.html", **_base_context())


@app.post("/analyze")
def analyze() -> tuple[str, int] | str:
    query = (request.form.get("query") or "").strip()
    ctx = _base_context(query)

    if not query:
        ctx["error"] = "Please enter a question before running the analysis."
        return render_template("index.html", **ctx), 400

    if not _auth_ready():
        ctx["error"] = (
            "Authentication not configured. "
            "Set GOOGLE_API_KEY or GOOGLE_APPLICATION_CREDENTIALS in your .env file."
        )
        return render_template("index.html", **ctx), 400

    try:
        result = asyncio.run(run_country_pipeline(query))
    except Exception as exc:
        ctx["error"] = f"Pipeline failed: {exc}"
        return render_template("index.html", **ctx), 500

    charts = [c for c in (_normalize_chart(j) for j in result.chart_jsons) if c]

    ctx.update({"result": result, "charts": charts})
    return render_template("index.html", **ctx)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
