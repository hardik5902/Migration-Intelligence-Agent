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
from tools.duckdb_tools import _ALLOWED_TABLES, _TTL_HOURS, _connect, db_path

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


@app.get("/cache")
def cache_viewer() -> str:
    """Browser-based DuckDB cache inspector."""
    from datetime import datetime, timezone, timedelta

    try:
        con = _connect()
        tables_info: list[dict[str, Any]] = []

        for table in sorted(_ALLOWED_TABLES):
            ttl_hours = _TTL_HOURS.get(table, 24)
            try:
                total = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except Exception:
                total = 0

            # Per-country summary
            try:
                rows = con.execute(
                    f"SELECT country, COUNT(*) as rows FROM {table} GROUP BY country ORDER BY country"
                ).fetchdf().to_dict(orient="records")
            except Exception:
                rows = []

            # Cache meta entries for this table
            try:
                meta = con.execute(
                    "SELECT cache_key, fetched_at FROM cache_meta WHERE table_name=? ORDER BY fetched_at DESC",
                    [table],
                ).fetchdf().to_dict(orient="records")
            except Exception:
                meta = []

            # Annotate each meta entry with TTL status
            now = datetime.now(timezone.utc)
            for m in meta:
                fetched = m.get("fetched_at")
                if fetched:
                    if isinstance(fetched, str):
                        try:
                            fetched = datetime.fromisoformat(fetched.replace("Z", "+00:00"))
                        except Exception:
                            fetched = None
                    if fetched:
                        if getattr(fetched, "tzinfo", None) is None:
                            fetched = fetched.replace(tzinfo=timezone.utc)
                        age = now - fetched
                        remaining = timedelta(hours=ttl_hours) - age
                        m["age_min"] = int(age.total_seconds() / 60)
                        m["valid"] = remaining.total_seconds() > 0
                        m["expires_in"] = (
                            f"{int(remaining.total_seconds()//3600)}h {int((remaining.total_seconds()%3600)//60)}m"
                            if m["valid"] else "expired"
                        )
                    else:
                        m["valid"] = False
                        m["expires_in"] = "unknown"
                        m["age_min"] = "?"
                else:
                    m["valid"] = False
                    m["expires_in"] = "unknown"
                    m["age_min"] = "?"

            tables_info.append({
                "name": table,
                "total_rows": total,
                "ttl_hours": ttl_hours,
                "country_rows": rows,
                "meta": meta,
            })

        # Sample rows for quick inspection (last table the user might query)
        samples: dict[str, list] = {}
        for table in sorted(_ALLOWED_TABLES):
            try:
                s = con.execute(f"SELECT * FROM {table} LIMIT 5").fetchdf().to_dict(orient="records")
                if s:
                    samples[table] = s
            except Exception:
                pass

        con.close()
        error = None
    except Exception as exc:
        tables_info = []
        samples = {}
        error = str(exc)

    return render_template(
        "cache.html",
        tables=tables_info,
        samples=samples,
        db_path=db_path(),
        error=error,
    )


@app.post("/cache/clear")
def cache_clear() -> Response:
    """Drop all rows from every cache table and reset cache_meta."""
    from flask import redirect, url_for
    try:
        con = _connect()
        for table in _ALLOWED_TABLES:
            con.execute(f"DELETE FROM {table}")
        con.execute("DELETE FROM cache_meta")
        con.close()
    except Exception:
        pass
    return redirect(url_for("cache_viewer"))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
