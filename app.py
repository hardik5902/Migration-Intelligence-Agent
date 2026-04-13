"""Flask frontend for the Migration Intelligence — Country Comparison pipeline."""

from __future__ import annotations

import json
import os
import queue
import threading
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from flask import Flask, Response, render_template, request, stream_with_context

from agents.country_pipeline import run_country_pipeline_streaming

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

app = Flask(__name__)

# Tool metadata shown in the UI
TOOL_CARDS = [
    {
        "id": "worldbank",
        "name": "World Bank",
        "icon": "📊",
        "description": "GDP growth, inflation, health & education spending, poverty, political stability",
    },
    {
        "id": "employment",
        "name": "ILO Labor",
        "icon": "💼",
        "description": "Unemployment rates, youth unemployment, labour force participation",
    },
    {
        "id": "unhcr",
        "name": "UNHCR",
        "icon": "🌍",
        "description": "Refugee displacement outflows and top destination countries",
    },
    {
        "id": "environment",
        "name": "Environment",
        "icon": "🌿",
        "description": "Climate trends (temperature, precipitation) + PM2.5 air quality — combined",
    },
    {
        "id": "acled",
        "name": "ACLED",
        "icon": "⚔️",
        "description": "Armed conflict events, fatalities, and crisis locations",
    },
    {
        "id": "news",
        "name": "News Summary",
        "icon": "📰",
        "description": "Recent news events summarised around your query topic per country",
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


def _sse(event_name: str, data: Any) -> str:
    return f"event: {event_name}\ndata: {json.dumps(data, default=str)}\n\n"


@app.get("/")
def index() -> str:
    return render_template("index.html", tool_cards=TOOL_CARDS)


@app.post("/analyze")
def analyze() -> Response:
    query = (request.form.get("query") or "").strip()

    if not query:
        def _err():
            yield _sse("error", {"message": "Please enter a question before running the analysis."})
        return Response(stream_with_context(_err()), content_type="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    if not _auth_ready():
        def _auth_err():
            yield _sse("error", {"message": (
                "Authentication not configured. "
                "Set GOOGLE_API_KEY or GOOGLE_APPLICATION_CREDENTIALS in your .env file."
            )})
        return Response(stream_with_context(_auth_err()), content_type="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    q: queue.Queue = queue.Queue()

    def emit(name: str, data: Any) -> None:
        q.put((name, data))

    def _run_pipeline() -> None:
        import asyncio as _asyncio
        try:
            _asyncio.run(run_country_pipeline_streaming(query, emit))
        except Exception as exc:
            q.put(("error", {"message": str(exc)}))
        finally:
            q.put(None)  # sentinel — tells the generator to stop

    threading.Thread(target=_run_pipeline, daemon=True).start()

    def generate():
        while True:
            try:
                item = q.get(timeout=300)
            except queue.Empty:
                yield _sse("error", {"message": "Pipeline timed out after 5 minutes."})
                break
            if item is None:
                break
            name, data = item
            yield _sse(name, data)

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
