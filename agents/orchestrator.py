"""Google ADK SequentialAgent pipeline: intent → scout → EDA → hypothesis."""

from __future__ import annotations

import os
from typing import Any

from google.adk.agents.llm_agent import Agent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

from agents.hypothesis_agent import build_hypothesis_agent
from agents.prompts import INTENT_CLASSIFIER_INSTRUCTION
from agents.workflow_agents import ParallelEdaAgent, ScoutDataAgent
from models.schemas import IntentConfig


def build_root_agent() -> SequentialAgent:
    intent_classifier = Agent(
        name="intent_classifier",
        model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        instruction=INTENT_CLASSIFIER_INSTRUCTION,
        output_schema=IntentConfig,
        output_key="intent_config",
        description="Classifies query intent and parameters.",
    )

    scout_data_agent = ScoutDataAgent(
        name="scout_data_agent",
        description="Fetches data from all available sources (World Bank, UNHCR, ACLED, Teleport, News, Climate, Employment, AQI).",
    )

    parallel_eda = ParallelEdaAgent(
        name="parallel_eda_agent",
        description="Runs EDA analyses (push factor, destination, pattern, relocation) with available data.",
    )

    hypothesis_agent = build_hypothesis_agent()

    return SequentialAgent(
        name="migration_pipeline",
        description="Migration intelligence pipeline: intent → collect all data → analyze → hypothesize.",
        sub_agents=[
            intent_classifier,
            scout_data_agent,
            parallel_eda,
            hypothesis_agent,
        ],
    )


def _compose_user_message(
    user_query: str,
    year_from: int | None,
    year_to: int | None,
) -> str:
    parts = [user_query.strip()]
    if year_from is not None and year_to is not None:
        parts.append(
            f"USE_YEAR_RANGE: set year_from={year_from} and year_to={year_to} in IntentConfig."
        )
    return "\n\n".join(parts)


async def run_migration_pipeline(
    user_query: str,
    user_id: str = "local_user",
    *,
    year_from: int | None = None,
    year_to: int | None = None,
) -> dict[str, Any]:
    """Execute the root SequentialAgent once and return session state snapshots."""
    # Check for either Vertex AI or direct API key authentication
    has_vertex_ai = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI") == "TRUE" and os.environ.get("GOOGLE_CLOUD_PROJECT")
    has_api_key = os.environ.get("GOOGLE_API_KEY")
    
    if not (has_vertex_ai or has_api_key):
        raise RuntimeError("Authentication required: set GOOGLE_CLOUD_PROJECT + GOOGLE_GENAI_USE_VERTEXAI (for Vertex AI) OR GOOGLE_API_KEY (for direct API). See .env.example")

    message = _compose_user_message(user_query, year_from, year_to)

    root = build_root_agent()
    session_service = InMemorySessionService()
    runner = Runner(
        app_name="migration_intel",
        agent=root,
        session_service=session_service,
        auto_create_session=True,
    )
    session_id = "session_main"
    async for _event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text=message)],
        ),
    ):
        pass

    session = await session_service.get_session(
        app_name="migration_intel",
        user_id=user_id,
        session_id=session_id,
    )
    if not session:
        return {}
    return dict(session.state)
