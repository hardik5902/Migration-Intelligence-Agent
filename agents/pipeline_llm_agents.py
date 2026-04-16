"""Native ADK LLM agents used by the country comparison pipeline."""

from __future__ import annotations

import os
from typing import Any

from google.adk.agents.llm_agent import Agent
from google.genai import types

from agents.analysis_agent import (
    UNIFIED_ANALYSIS_INSTRUCTION,
    build_unified_analysis_prompt,
)
from agents.tool_selector import TOOL_SELECTOR_INSTRUCTION, build_tool_selector_prompt
from models.schemas import ToolSelectorOutput, UnifiedAnalysisOutput
from tools.artifact_store import read_json_artifact


def _model_name() -> str:
    return os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")


def _load_state_payload(
    state: dict[str, Any],
    *,
    key: str,
    ref_key: str,
    default: Any,
) -> Any:
    if ref := state.get(ref_key):
        payload = read_json_artifact(str(ref))
        if payload is not None:
            return payload
    return state.get(key, default)


def _prepare_tool_selector_request(ctx: Any, llm_request: Any) -> None:
    query = str(ctx.session.state.get("user_query", "")).strip()
    llm_request.contents = [
        types.Content(
            role="user",
            parts=[types.Part(text=build_tool_selector_prompt(query))],
        )
    ]
    llm_request.config.response_mime_type = "application/json"


def _prepare_unified_analysis_request(ctx: Any, llm_request: Any) -> None:
    state = ctx.session.state
    selector = ToolSelectorOutput.model_validate(state.get("selector_output", {}))
    prompt, _, _ = build_unified_analysis_prompt(
        query=str(state.get("user_query", "")),
        query_focus=selector.query_focus,
        selected_tools=selector.selected_tools,
        countries=selector.countries,
        comparison_manifest=_load_state_payload(
            state,
            key="comparison_manifest",
            ref_key="comparison_manifest_ref",
            default=[],
        ),
        eda_chart_manifest=_load_state_payload(
            state,
            key="eda_chart_manifest",
            ref_key="eda_chart_manifest_ref",
            default=[],
        ),
        eda_findings=_load_state_payload(
            state,
            key="eda_findings",
            ref_key="eda_findings_ref",
            default={},
        ),
        data_coverage=_load_state_payload(
            state,
            key="data_coverage",
            ref_key="data_coverage_ref",
            default={},
        ),
    )
    llm_request.contents = [
        types.Content(
            role="user",
            parts=[types.Part(text=prompt)],
        )
    ]
    llm_request.config.response_mime_type = "application/json"


def build_tool_selector_llm_agent() -> Agent:
    return Agent(
        name="tool_selector_llm",
        description="LLM selects relevant tools and candidate countries for the query.",
        model=_model_name(),
        instruction=TOOL_SELECTOR_INSTRUCTION,
        output_schema=ToolSelectorOutput,
        output_key="tool_selector_raw",
        include_contents="none",
        generate_content_config=types.GenerateContentConfig(temperature=0.1),
        before_model_callback=_prepare_tool_selector_request,
    )


def build_unified_analysis_llm_agent() -> Agent:
    return Agent(
        name="unified_analysis_llm",
        description="LLM selects chart keys and writes hypotheses from EDA artifacts.",
        model=_model_name(),
        instruction=UNIFIED_ANALYSIS_INSTRUCTION,
        output_schema=UnifiedAnalysisOutput,
        output_key="unified_analysis_raw",
        include_contents="none",
        generate_content_config=types.GenerateContentConfig(temperature=0.2),
        before_model_callback=_prepare_unified_analysis_request,
    )
