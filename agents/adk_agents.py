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

from agents.data_collector import collect_data_for_countries
from agents.eda_analyst import run_eda
from agents.evidence_generator import generate_hypotheses
from agents.pipeline_config import TOOL_DATA_KEYS
from agents.tool_selector import analyze_query_with_llm
from analysis.country_charts import build_country_comparison_charts
from analysis.eda_charts import build_eda_charts
from models.schemas import HypothesisInsight, ToolSelectorOutput

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

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            actions=EventActions(
                state_delta={"countries_data": countries_data}
            ),
        )


# ---------------------------------------------------------------------------
# Agent 3 — Exploratory Data Analysis
# ---------------------------------------------------------------------------

class EDAAnalystADKAgent(BaseAgent):
    """Runs statistical EDA (growth rates, anomalies, correlations) on collected data."""

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
            "message": "Running growth-rate, anomaly, and correlation analysis…",
        })

        # run_eda calls run_growth_rate, run_anomaly_detect, run_correlation_analysis
        eda_findings = await asyncio.to_thread(
            run_eda,
            countries_data,
            selector.selected_tools,
            selector.query_focus,
            selector.worldbank_indicators,
        )

        # Build up to 2 EDA-specific charts (heatmap / CAGR bar / anomaly / spread)
        eda_chart_jsons: list[str] = await asyncio.to_thread(
            build_eda_charts,
            eda_findings,
            countries_data,
        )

        from plotly.io import from_json as _plotly_from_json
        eda_charts_normalized: list[Any] = []
        for j in eda_chart_jsons:
            try:
                eda_charts_normalized.append(_plotly_from_json(j).to_plotly_json())
            except Exception:
                pass

        _emit("eda", {
            "findings":   eda_findings.get("findings", []),
            "eda_charts": eda_charts_normalized,
            "correlations": eda_findings.get("correlations", []),
        })

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            actions=EventActions(
                state_delta={
                    "eda_findings":   eda_findings,
                    "eda_charts":     eda_charts_normalized,
                }
            ),
        )


# ---------------------------------------------------------------------------
# Agent 4 — Chart building + evidence generation (runs both concurrently)
# ---------------------------------------------------------------------------

class ChartEvidenceADKAgent(BaseAgent):
    """Builds 4 Plotly charts and 3 hypothesis insights concurrently; emits both."""

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
        countries_data: dict[str, Any] = ctx.session.state.get("countries_data", {})
        eda_findings: dict[str, Any] = ctx.session.state.get("eda_findings", {})
        query = pipeline_query.get() or ctx.session.state.get("user_query", "")

        _emit("stage", {
            "step": 4, "total": 5,
            "label": "Building charts & insights",
            "message": (
                "Generating 4 comparison charts and "
                "3 evidence insights in parallel…"
            ),
        })

        # Run chart building (sync/CPU) and hypothesis LLM call concurrently
        chart_jsons, hypotheses = await asyncio.gather(
            asyncio.to_thread(
                build_country_comparison_charts,
                countries_data,
                selector.selected_tools,
                selector.query_focus,
                selector.worldbank_indicators,   # prioritise query-relevant charts
            ),
            generate_hypotheses(query, countries_data, selector.selected_tools, eda_findings, selector.query_focus),
        )

        # Normalise Plotly JSON strings → dicts for the browser JS client
        charts_normalized: list[Any] = []
        for j in chart_jsons:
            try:
                charts_normalized.append(_plotly_from_json(j).to_plotly_json())
            except Exception:
                pass

        _emit("charts", {"charts": charts_normalized})

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
                    "charts":      charts_normalized,
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
