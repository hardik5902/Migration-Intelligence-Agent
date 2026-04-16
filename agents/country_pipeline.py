"""Country comparison pipeline — ADK-powered orchestrator.

Wires the four ADK agents (ToolSelectorADKAgent, DataCollectorADKAgent,
EDAAnalystADKAgent, UnifiedAnalysisADKAgent) through ADK's Runner +
InMemorySessionService.

Data flows through ADK session state:
  user_query          → set before runner starts
  selector_output     ← ToolSelectorADKAgent
  countries_data      ← DataCollectorADKAgent
  eda_findings        ← EDAAnalystADKAgent
  data_coverage       ← EDAAnalystADKAgent
  comparison_manifest ← EDAAnalystADKAgent
  eda_chart_manifest  ← EDAAnalystADKAgent
  charts              ← UnifiedAnalysisADKAgent
  eda_charts          ← UnifiedAnalysisADKAgent
  hypotheses          ← UnifiedAnalysisADKAgent
  tool_stats          ← UnifiedAnalysisADKAgent

SSE events are fired directly by each agent via the pipeline_emit contextvar,
so the Flask SSE generator receives stage/selection/charts/evidence/done events
progressively as the pipeline advances.

Two public entry points:
  run_country_pipeline_streaming(query, emit) — for the Flask SSE route
  run_country_pipeline(query)               — returns CountryComparisonResult
"""

from __future__ import annotations

import uuid
from typing import Any, Callable

from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

from agents.adk_agents import (
    build_country_comparison_pipeline,
    pipeline_emit,
    pipeline_query,
)
from agents.progress_tracker import reset_tracker
from models.schemas import CountryComparisonResult, HypothesisInsight
from tools.artifact_store import read_json_artifact

_APP_NAME = "migration_intel"
_USER_ID = "web_user"


async def _run_adk_pipeline(query: str) -> dict[str, Any]:
    """Create a fresh Runner, execute the SequentialAgent, return session state."""
    pipeline = build_country_comparison_pipeline()
    session_service = InMemorySessionService()

    session_id = f"sess_{uuid.uuid4().hex[:10]}"

    # Pre-populate session state with the user query so BaseAgents can read it
    await session_service.create_session(
        app_name=_APP_NAME,
        user_id=_USER_ID,
        session_id=session_id,
        state={"user_query": query},
    )

    runner = Runner(
        app_name=_APP_NAME,
        agent=pipeline,
        session_service=session_service,
        auto_create_session=False,
    )

    async for _event in runner.run_async(
        user_id=_USER_ID,
        session_id=session_id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text=query)],
        ),
    ):
        pass  # agents fire SSE events themselves via pipeline_emit / pipeline_query

    session = await session_service.get_session(
        app_name=_APP_NAME,
        user_id=_USER_ID,
        session_id=session_id,
    )
    return dict(session.state) if session else {}


# ---------------------------------------------------------------------------
# SSE entry point (Flask route)
# ---------------------------------------------------------------------------

async def run_country_pipeline_streaming(
    query: str,
    emit: Callable[[str, Any], None],
) -> None:
    """Execute the ADK pipeline and stream events to the Flask SSE generator."""
    reset_tracker()

    # Inject emit + query into contextvars so agents can reach them
    pipeline_emit.set(emit)
    pipeline_query.set(query)

    try:
        await _run_adk_pipeline(query)
    except Exception as exc:
        emit("error", {"message": f"Pipeline failed: {exc}"})
        return

    emit("done", {"message": "Analysis complete."})


# ---------------------------------------------------------------------------
# Non-streaming entry point (CLI / tests)
# ---------------------------------------------------------------------------

async def run_country_pipeline(query: str) -> CountryComparisonResult:
    """Run the ADK pipeline and return a structured result (no SSE)."""
    reset_tracker()

    # No-op emit so agents don't crash when calling _emit()
    pipeline_emit.set(lambda *_: None)
    pipeline_query.set(query)

    state = await _run_adk_pipeline(query)

    import json as _json

    selector_data = state.get("selector_output") or {}
    hypotheses_raw = state.get("hypotheses") or []
    chart_jsons: list[str] = []

    chart_payload = state.get("charts") or []
    if state.get("charts_ref"):
        chart_payload = read_json_artifact(str(state["charts_ref"])) or []

    # Re-serialise charts to JSON strings for CountryComparisonResult.chart_jsons
    for chart_dict in chart_payload:
        try:
            chart_jsons.append(_json.dumps(chart_dict))
        except Exception:
            pass

    hypotheses = []
    for h in hypotheses_raw:
        try:
            hypotheses.append(HypothesisInsight.model_validate(h))
        except Exception:
            pass

    return CountryComparisonResult(
        query=query,
        countries=selector_data.get("countries", []),
        country_codes=selector_data.get("country_codes", []),
        tools_used=selector_data.get("selected_tools", []),
        hypotheses=hypotheses,
        summary=selector_data.get("reasoning", ""),
        chart_jsons=chart_jsons,
        query_focus=selector_data.get("query_focus", ""),
        year_from=selector_data.get("year_from", 2015),
        year_to=selector_data.get("year_to", 2023),
        tool_stats=state.get("tool_stats") or {},
    )
