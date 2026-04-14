"""ADK BaseAgent implementations for the country comparison pipeline.

Three agents run in sequence via ADK's SequentialAgent + Runner:
  1. ToolSelectorADKAgent   — LLM picks tools + K countries
  2. DataCollectorADKAgent  — parallel live-data fetch for every country
  3. ChartEvidenceADKAgent  — chart building + evidence generation (concurrent)

Session state is the data bus:
  • user_query        (str)           — set before the runner starts
  • selector_output   (dict)          — written by ToolSelectorADKAgent
  • countries_data    (dict)          — written by DataCollectorADKAgent
  • charts            (list)          — written by ChartEvidenceADKAgent
  • hypotheses        (list)          — written by ChartEvidenceADKAgent
  • tool_stats        (dict)          — written by ChartEvidenceADKAgent

The SSE emit callback and the query are stored in contextvars so every agent
can access them without needing them threaded through constructors.
"""

from __future__ import annotations

import asyncio
import contextvars
from typing import Any, AsyncGenerator, Callable

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from typing_extensions import override

from agents.analysis_agent import build_data_coverage_report, run_unified_analysis
from agents.data_collector import collect_data_for_countries, filter_countries_by_coverage
from agents.eda_analyst import run_eda
from agents.pipeline_config import TOOL_DATA_KEYS
from agents.tool_selector import analyze_query_with_llm
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
from models.schemas import ToolSelectorOutput


# ---------------------------------------------------------------------------
# Context vars — set once per pipeline run; readable by every agent coroutine
# ---------------------------------------------------------------------------

pipeline_emit: contextvars.ContextVar[Callable[[str, Any], None]] = (
    contextvars.ContextVar("pipeline_emit", default=lambda *_: None)
)

pipeline_query: contextvars.ContextVar[str] = (
    contextvars.ContextVar("pipeline_query", default="")
)


def _emit(name: str, data: Any) -> None:
    """Convenience wrapper — calls whatever emit fn is active for this run."""
    pipeline_emit.get()(name, data)


# ---------------------------------------------------------------------------
# Agent 1 — Tool + country selection
# ---------------------------------------------------------------------------

class ToolSelectorADKAgent(BaseAgent):
    """Calls the LLM to pick tools + K countries; writes selector_output."""

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        _emit("stage", {
            "step": 1, "total": 4,
            "label": "AI selects tools & countries",
            "message": "Analysing your query…",
        })

        query = pipeline_query.get() or ctx.session.state.get("user_query", "")
        selector = await analyze_query_with_llm(query)

        # If LLM determined query is out of scope, stop the pipeline immediately
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
                actions=EventActions(state_delta={"out_of_scope": True}),
            )
            return

        _emit("selection", {
            "countries":     selector.countries,
            "country_codes": selector.country_codes,
            "tools":         selector.selected_tools,
            "year_from":     selector.year_from,
            "year_to":       selector.year_to,
            "query_focus":   selector.query_focus,
            "proxy_note":    selector.proxy_note,
            "country_strategy": selector.country_strategy,
            "k":             selector.k,
        })

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            actions=EventActions(
                state_delta={"selector_output": selector.model_dump()}
            ),
        )


# ---------------------------------------------------------------------------
# Agent 2 — Parallel data collection
# ---------------------------------------------------------------------------

class DataCollectorADKAgent(BaseAgent):
    """Reads selector_output, fetches live data for all countries in parallel."""

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        if ctx.session.state.get("out_of_scope"):
            return  # pipeline was stopped by ToolSelectorADKAgent

        selector = ToolSelectorOutput.model_validate(
            ctx.session.state.get("selector_output", {})
        )

        _emit("stage", {
            "step": 2, "total": 4,
            "label": "Collecting live data",
            "message": (
                f"Fetching data for {len(selector.countries)} "
                "countries in parallel…"
            ),
        })

        countries_data = await collect_data_for_countries(
            selector.countries,
            selector.country_codes,
            selector.selected_tools,
            selector.year_from,
            selector.year_to,
        )

        (
            countries_data,
            validated_countries,
            validated_codes,
            coverage_ranking,
        ) = filter_countries_by_coverage(
            countries_data,
            selector.country_codes,
            selector.selected_tools,
            selector.k,
        )

        selector.countries = validated_countries
        selector.country_codes = validated_codes
        selector.k = len(validated_countries)

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
                    "countries_data": countries_data,
                    "selector_output": selector.model_dump(),
                    "coverage_ranking": coverage_ranking,
                }
            ),
        )


# ---------------------------------------------------------------------------
# Agent 3 — Exploratory Data Analysis
# ---------------------------------------------------------------------------

class EDAAnalystADKAgent(BaseAgent):
    """Runs statistical EDA, builds data-coverage report + comparison/EDA chart manifests.

    This agent is pure Python — no LLM calls. It prepares everything the
    unified analysis agent needs in a single LLM request.
    """

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        if ctx.session.state.get("out_of_scope"):
            return

        selector = ToolSelectorOutput.model_validate(
            ctx.session.state.get("selector_output", {})
        )
        countries_data: dict[str, Any] = ctx.session.state.get("countries_data", {})

        _emit("stage", {
            "step": 3, "total": 5,
            "label": "Exploratory data analysis",
            "message": "Running stats, decoding data coverage, building chart manifests…",
        })

        # Pure-Python stats (CAGR / anomaly / correlation / summary)
        eda_findings = await asyncio.to_thread(
            run_eda,
            countries_data,
            selector.selected_tools,
            selector.query_focus,
            selector.worldbank_indicators,
        )

        # Data-coverage introspection: decode what columns came back per country
        data_coverage = await asyncio.to_thread(
            build_data_coverage_report,
            countries_data,
            selector.selected_tools,
        )

        # Comparison chart manifest (from country_charts registry)
        comparison_manifest = await asyncio.to_thread(
            build_registry_manifest, countries_data
        )

        # EDA chart manifest (which stat charts can actually render)
        eda_chart_manifest = await asyncio.to_thread(
            build_eda_chart_manifest, eda_findings, countries_data
        )

        _emit("eda", {
            "findings":     eda_findings.get("findings", []),
            "eda_charts":   [],  # charts come later via the 'eda_charts' event
            "correlations": eda_findings.get("correlations", []),
        })

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            actions=EventActions(
                state_delta={
                    "eda_findings":        eda_findings,
                    "data_coverage":       data_coverage,
                    "comparison_manifest": comparison_manifest,
                    "eda_chart_manifest":  eda_chart_manifest,
                }
            ),
        )


# ---------------------------------------------------------------------------
# Agent 4 — Unified analysis: single LLM call returns everything
# ---------------------------------------------------------------------------

class UnifiedAnalysisADKAgent(BaseAgent):
    """Single LLM call: picks comparison charts, EDA charts, and writes hypotheses."""

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        if ctx.session.state.get("out_of_scope"):
            return

        from plotly.io import from_json as _plotly_from_json

        selector = ToolSelectorOutput.model_validate(
            ctx.session.state.get("selector_output", {})
        )
        countries_data:      dict[str, Any]  = ctx.session.state.get("countries_data", {})
        eda_findings:        dict[str, Any]  = ctx.session.state.get("eda_findings", {})
        data_coverage:       dict[str, Any]  = ctx.session.state.get("data_coverage", {})
        comparison_manifest: list[dict]      = ctx.session.state.get("comparison_manifest", [])
        eda_chart_manifest:  list[dict]      = ctx.session.state.get("eda_chart_manifest", [])
        query = pipeline_query.get() or ctx.session.state.get("user_query", "")

        _emit("stage", {
            "step": 4, "total": 5,
            "label": "Building charts & insights",
            "message": "Single LLM call: picking charts + writing hypotheses…",
        })

        # --- One LLM call: chart keys (comparison + EDA) + hypotheses ---
        result = await run_unified_analysis(
            query=query,
            query_focus=selector.query_focus,
            selected_tools=selector.selected_tools,
            countries=selector.countries,
            comparison_manifest=comparison_manifest,
            eda_chart_manifest=eda_chart_manifest,
            eda_findings=eda_findings,
            data_coverage=data_coverage,
        )

        comparison_keys = result["comparison_chart_keys"]
        eda_keys        = result["eda_chart_keys"]
        hypotheses      = result["hypotheses"]

        # --- Render comparison charts (4) ---
        if len(comparison_keys) >= 4:
            comp_chart_jsons = await asyncio.to_thread(
                render_charts_by_keys,
                comparison_keys,
                comparison_manifest,
                countries_data,
            )
        else:
            # Fallback to heuristic builder if LLM returned < 4 valid keys
            comp_chart_jsons = await asyncio.to_thread(
                build_country_comparison_charts,
                countries_data,
                selector.selected_tools,
                selector.query_focus,
                selector.worldbank_indicators,
            )

        # --- Render EDA charts (up to 4) ---
        if eda_keys:
            eda_chart_jsons = await asyncio.to_thread(
                render_eda_charts_by_keys,
                eda_keys,
                eda_findings,
                countries_data,
            )
        else:
            # Fallback: heuristic picks up to 4 from the manifest
            eda_chart_jsons = await asyncio.to_thread(
                build_eda_charts,
                eda_findings,
                countries_data,
                4,
            )

        # Normalise Plotly JSON → dict for the browser
        def _jsons_to_dicts(chart_jsons: list[str]) -> list[Any]:
            result: list[Any] = []
            for j in chart_jsons:
                try:
                    result.append(_plotly_from_json(j).to_plotly_json())
                except Exception:
                    pass
            return result

        comp_charts_normalized = _jsons_to_dicts(comp_chart_jsons)
        eda_charts_normalized  = _jsons_to_dicts(eda_chart_jsons)

        # Emit EDA charts (findings were emitted earlier by EDAAnalystADKAgent)
        if eda_charts_normalized:
            _emit("eda_charts", {"eda_charts": eda_charts_normalized})

        _emit("charts", {"charts": comp_charts_normalized})

        # Per-tool row counts for the status badges
        tool_stats: dict[str, int] = {
            tool: sum(
                len(countries_data[country].get(k) or [])
                for country in countries_data
                for k in TOOL_DATA_KEYS.get(tool, [tool])
            )
            for tool in selector.selected_tools
        }

        _emit("stage", {
            "step": 5, "total": 5,
            "label": "Finalising results",
            "message": "Wrapping up…",
        })

        _emit("evidence", {
            "hypotheses":  [h.model_dump() for h in hypotheses],
            "tool_stats":  tool_stats,
            "countries":   selector.countries,
            "tools_used":  selector.selected_tools,
            "year_from":   selector.year_from,
            "year_to":     selector.year_to,
            "query_focus": selector.query_focus,
        })

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            actions=EventActions(
                state_delta={
                    "charts":      comp_charts_normalized,
                    "eda_charts":  eda_charts_normalized,
                    "hypotheses":  [h.model_dump() for h in hypotheses],
                    "tool_stats":  tool_stats,
                }
            ),
        )


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def build_country_comparison_pipeline() -> SequentialAgent:
    """Assemble the three agents into an ADK SequentialAgent."""
    return SequentialAgent(
        name="country_comparison_pipeline",
        description=(
            "Country comparison pipeline: "
            "select tools → collect data → EDA → build charts + insights"
        ),
        sub_agents=[
            ToolSelectorADKAgent(
                name="tool_selector",
                description="LLM selects relevant tools and K countries for the query.",
            ),
            DataCollectorADKAgent(
                name="data_collector",
                description="Fetches live data for all selected countries in parallel.",
            ),
            EDAAnalystADKAgent(
                name="eda_analyst",
                description=(
                    "Runs statistical EDA: growth rates (CAGR), anomaly detection, "
                    "and cross-country correlations. Stores eda_findings + eda_charts."
                ),
            ),
            ChartEvidenceADKAgent(
                name="chart_evidence",
                description=(
                    "Builds 4 comparison charts and 3 evidence insights "
                    "informed by EDA findings."
                ),
            ),
        ],
    )
