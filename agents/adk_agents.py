"""ADK agents for the country comparison pipeline."""

from __future__ import annotations

import asyncio
import contextvars
from typing import Any, AsyncGenerator, Callable

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from plotly.io import from_json as _plotly_from_json
from typing_extensions import override

from agents.analysis_agent import (
    build_data_coverage_report,
    validate_unified_analysis_output,
)
from agents.data_collector import collect_data_for_countries, filter_countries_by_coverage
from agents.eda_analyst import run_eda
from agents.pipeline_config import TOOL_DATA_KEYS
from agents.pipeline_llm_agents import (
    build_tool_selector_llm_agent,
    build_unified_analysis_llm_agent,
)
from agents.tool_selector import normalize_tool_selector_output
from analysis.country_charts import (
    build_country_comparison_charts,
    build_registry_manifest,
    render_charts_by_keys,
)
from analysis.eda_charts import (
    build_eda_chart_manifest,
    build_eda_charts,
    render_eda_charts_by_keys,
)
from models.schemas import ToolSelectorOutput, UnifiedAnalysisOutput
from tools.artifact_store import read_json_artifact, write_json_artifact

pipeline_emit: contextvars.ContextVar[Callable[[str, Any], None]] = (
    contextvars.ContextVar("pipeline_emit", default=lambda *_: None)
)
pipeline_query: contextvars.ContextVar[str] = (
    contextvars.ContextVar("pipeline_query", default="")
)


def _emit(name: str, data: Any) -> None:
    pipeline_emit.get()(name, data)


def _store_state_artifact(session_id: str, name: str, payload: Any) -> str:
    return write_json_artifact(session_id, name, payload)


def _load_state_payload(
    ctx: InvocationContext,
    *,
    key: str,
    ref_key: str,
    default: Any,
) -> Any:
    if ref := ctx.session.state.get(ref_key):
        payload = read_json_artifact(str(ref))
        if payload is not None:
            return payload
    return ctx.session.state.get(key, default)


def _noop_event(ctx: InvocationContext, author: str) -> Event:
    return Event(
        invocation_id=ctx.invocation_id,
        author=author,
        branch=ctx.branch,
        actions=EventActions(),
    )


class ToolSelectorStageADKAgent(BaseAgent):
    """Emits the stage update before the selector LLM runs."""

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        _emit("stage", {
            "step": 1,
            "total": 5,
            "label": "AI selects tools & countries",
            "message": "Analysing your query...",
        })
        yield _noop_event(ctx, self.name)


class ToolSelectorResultADKAgent(BaseAgent):
    """Normalises structured selector output and emits the chosen scope."""

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        query = pipeline_query.get() or ctx.session.state.get("user_query", "")
        selector = normalize_tool_selector_output(
            query,
            ctx.session.state.get("tool_selector_raw", {}),
        )

        if not selector.in_scope:
            _emit("out_of_scope", {
                "reason": selector.out_of_scope_reason or (
                    "This query doesn't relate to countries, migration, or quality-of-life "
                    "metrics that our data sources can answer."
                ),
            })
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                branch=ctx.branch,
                actions=EventActions(
                    state_delta={
                        "out_of_scope": True,
                        "selector_output": selector.model_dump(),
                    }
                ),
            )
            return

        _emit("selection", {
            "countries": selector.countries,
            "country_codes": selector.country_codes,
            "tools": selector.selected_tools,
            "year_from": selector.year_from,
            "year_to": selector.year_to,
            "query_focus": selector.query_focus,
            "proxy_note": selector.proxy_note,
            "country_strategy": selector.country_strategy,
            "k": selector.k,
        })
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            actions=EventActions(state_delta={"selector_output": selector.model_dump()}),
        )


class DataCollectorADKAgent(BaseAgent):
    """Reads selector_output, fetches live data, and stores bulky data as artifacts."""

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        if ctx.session.state.get("out_of_scope"):
            return

        selector = ToolSelectorOutput.model_validate(ctx.session.state.get("selector_output", {}))
        _emit("stage", {
            "step": 2,
            "total": 5,
            "label": "Collecting live data",
            "message": f"Fetching data for {len(selector.countries)} countries in parallel...",
        })

        countries_data = await collect_data_for_countries(
            selector.countries,
            selector.country_codes,
            selector.selected_tools,
            selector.year_from,
            selector.year_to,
        )
        countries_data, validated_countries, validated_codes, coverage_ranking = (
            filter_countries_by_coverage(
                countries_data,
                selector.country_codes,
                selector.selected_tools,
                selector.k,
            )
        )
        selector.countries = validated_countries
        selector.country_codes = validated_codes
        selector.k = len(validated_countries)

        countries_data_ref = await asyncio.to_thread(
            _store_state_artifact,
            ctx.session.id,
            "countries_data",
            countries_data,
        )

        _emit("selection", {
            "countries": selector.countries,
            "country_codes": selector.country_codes,
            "tools": selector.selected_tools,
            "year_from": selector.year_from,
            "year_to": selector.year_to,
            "query_focus": selector.query_focus,
            "proxy_note": selector.proxy_note,
            "k": selector.k,
            "country_strategy": selector.country_strategy,
            "coverage_ranking": coverage_ranking,
        })

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            actions=EventActions(
                state_delta={
                    "countries_data_ref": countries_data_ref,
                    "selector_output": selector.model_dump(),
                    "coverage_ranking": coverage_ranking,
                }
            ),
        )


class EDAAnalystADKAgent(BaseAgent):
    """Runs pure-Python EDA and stores large outputs as artifacts."""

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        if ctx.session.state.get("out_of_scope"):
            return

        selector = ToolSelectorOutput.model_validate(ctx.session.state.get("selector_output", {}))
        countries_data = _load_state_payload(
            ctx,
            key="countries_data",
            ref_key="countries_data_ref",
            default={},
        )

        _emit("stage", {
            "step": 3,
            "total": 5,
            "label": "Exploratory data analysis",
            "message": "Running stats, decoding data coverage, building chart manifests...",
        })

        eda_findings = await asyncio.to_thread(
            run_eda,
            countries_data,
            selector.selected_tools,
            selector.query_focus,
            selector.worldbank_indicators,
        )
        data_coverage = await asyncio.to_thread(
            build_data_coverage_report,
            countries_data,
            selector.selected_tools,
        )
        comparison_manifest = await asyncio.to_thread(build_registry_manifest, countries_data)
        eda_chart_manifest = await asyncio.to_thread(
            build_eda_chart_manifest,
            eda_findings,
            countries_data,
        )

        eda_findings_ref = await asyncio.to_thread(
            _store_state_artifact,
            ctx.session.id,
            "eda_findings",
            eda_findings,
        )
        data_coverage_ref = await asyncio.to_thread(
            _store_state_artifact,
            ctx.session.id,
            "data_coverage",
            data_coverage,
        )
        comparison_manifest_ref = await asyncio.to_thread(
            _store_state_artifact,
            ctx.session.id,
            "comparison_manifest",
            comparison_manifest,
        )
        eda_chart_manifest_ref = await asyncio.to_thread(
            _store_state_artifact,
            ctx.session.id,
            "eda_chart_manifest",
            eda_chart_manifest,
        )

        _emit("eda", {
            "findings": eda_findings.get("findings", []),
            "eda_charts": [],
            "correlations": eda_findings.get("correlations", []),
        })

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            actions=EventActions(
                state_delta={
                    "eda_findings_ref": eda_findings_ref,
                    "data_coverage_ref": data_coverage_ref,
                    "comparison_manifest_ref": comparison_manifest_ref,
                    "eda_chart_manifest_ref": eda_chart_manifest_ref,
                }
            ),
        )


class UnifiedAnalysisStageADKAgent(BaseAgent):
    """Emits the stage update before the unified analysis LLM runs."""

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        if ctx.session.state.get("out_of_scope"):
            return
        _emit("stage", {
            "step": 4,
            "total": 5,
            "label": "Building charts & insights",
            "message": "LLM is selecting charts and drafting hypotheses...",
        })
        yield _noop_event(ctx, self.name)


class UnifiedAnalysisADKAgent(BaseAgent):
    """Validates structured LLM output, renders charts, and emits final results."""

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        if ctx.session.state.get("out_of_scope"):
            return

        selector = ToolSelectorOutput.model_validate(ctx.session.state.get("selector_output", {}))
        countries_data = _load_state_payload(
            ctx,
            key="countries_data",
            ref_key="countries_data_ref",
            default={},
        )
        eda_findings = _load_state_payload(
            ctx,
            key="eda_findings",
            ref_key="eda_findings_ref",
            default={},
        )
        comparison_manifest = _load_state_payload(
            ctx,
            key="comparison_manifest",
            ref_key="comparison_manifest_ref",
            default=[],
        )
        eda_chart_manifest = _load_state_payload(
            ctx,
            key="eda_chart_manifest",
            ref_key="eda_chart_manifest_ref",
            default=[],
        )
        raw_output = UnifiedAnalysisOutput.model_validate(
            ctx.session.state.get("unified_analysis_raw", {})
        )

        result = validate_unified_analysis_output(
            output=raw_output,
            comparison_manifest=[
                manifest
                for manifest in comparison_manifest
                if len(manifest.get("countries_with_data", [])) >= 2
            ],
            eda_chart_manifest=eda_chart_manifest,
            selected_tools=selector.selected_tools,
        )
        comparison_keys = result["comparison_chart_keys"]
        eda_keys = result["eda_chart_keys"]
        hypotheses = result["hypotheses"]

        if comparison_keys:
            comp_chart_jsons = await asyncio.to_thread(
                render_charts_by_keys,
                comparison_keys,
                comparison_manifest,
                countries_data,
            )
        else:
            comp_chart_jsons = await asyncio.to_thread(
                build_country_comparison_charts,
                countries_data,
                selector.selected_tools,
                selector.query_focus,
                selector.worldbank_indicators,
            )

        if eda_keys:
            eda_chart_jsons = await asyncio.to_thread(
                render_eda_charts_by_keys,
                eda_keys,
                eda_findings,
                countries_data,
            )
        else:
            eda_chart_jsons = await asyncio.to_thread(
                build_eda_charts,
                eda_findings,
                countries_data,
                4,
            )

        def _jsons_to_dicts(chart_jsons: list[str]) -> list[Any]:
            result_dicts: list[Any] = []
            for chart_json in chart_jsons:
                try:
                    result_dicts.append(_plotly_from_json(chart_json).to_plotly_json())
                except Exception:
                    pass
            return result_dicts

        comp_charts_normalized = _jsons_to_dicts(comp_chart_jsons)
        eda_charts_normalized = _jsons_to_dicts(eda_chart_jsons)

        charts_ref = await asyncio.to_thread(
            _store_state_artifact,
            ctx.session.id,
            "charts",
            comp_charts_normalized,
        )
        eda_charts_ref = await asyncio.to_thread(
            _store_state_artifact,
            ctx.session.id,
            "eda_charts",
            eda_charts_normalized,
        )

        if eda_charts_normalized:
            _emit("eda_charts", {"eda_charts": eda_charts_normalized})
        _emit("charts", {"charts": comp_charts_normalized})

        tool_stats: dict[str, int] = {
            tool: sum(
                len(countries_data[country].get(data_key) or [])
                for country in countries_data
                for data_key in TOOL_DATA_KEYS.get(tool, [tool])
            )
            for tool in selector.selected_tools
        }

        _emit("stage", {
            "step": 5,
            "total": 5,
            "label": "Finalising results",
            "message": "Wrapping up...",
        })
        _emit("evidence", {
            "hypotheses": [hypothesis.model_dump() for hypothesis in hypotheses],
            "tool_stats": tool_stats,
            "countries": selector.countries,
            "tools_used": selector.selected_tools,
            "year_from": selector.year_from,
            "year_to": selector.year_to,
            "query_focus": selector.query_focus,
        })

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            actions=EventActions(
                state_delta={
                    "charts_ref": charts_ref,
                    "eda_charts_ref": eda_charts_ref,
                    "hypotheses": [hypothesis.model_dump() for hypothesis in hypotheses],
                    "tool_stats": tool_stats,
                }
            ),
        )


def build_country_comparison_pipeline() -> SequentialAgent:
    """Assemble the country-comparison pipeline."""
    return SequentialAgent(
        name="country_comparison_pipeline",
        description=(
            "Country comparison pipeline: select tools, collect data, run EDA, "
            "then build charts and insights."
        ),
        sub_agents=[
            ToolSelectorStageADKAgent(
                name="tool_selector_stage",
                description="Emits the tool-selection stage update.",
            ),
            build_tool_selector_llm_agent(),
            ToolSelectorResultADKAgent(
                name="tool_selector_result",
                description="Normalises the selector output and emits the chosen scope.",
            ),
            DataCollectorADKAgent(
                name="data_collector",
                description="Fetches live data for all selected countries in parallel.",
            ),
            EDAAnalystADKAgent(
                name="eda_analyst",
                description="Runs statistical EDA and stores artifacts for downstream analysis.",
            ),
            UnifiedAnalysisStageADKAgent(
                name="unified_analysis_stage",
                description="Emits the unified-analysis stage update.",
            ),
            build_unified_analysis_llm_agent(),
            UnifiedAnalysisADKAgent(
                name="unified_analysis",
                description="Validates structured output, renders charts, and emits final results.",
            ),
        ],
    )
