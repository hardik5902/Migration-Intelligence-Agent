"""Non-LLM workflow steps as BaseAgent (scout + parallel EDA)."""

from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator

from typing_extensions import override

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions

from agents.destination_tracker import run_destination_tracker
from agents.pattern_detective import run_pattern_detective
from agents.push_factor_analyst import run_push_factor_analysis
from agents.relocation_scorer import run_relocation_scorer
from agents.progress_tracker import get_tracker
from agents.scout_service import collect_migration_dataset
from models.schemas import IntentConfig, MigrationDataset


def _parse_intent(raw: object) -> IntentConfig:
    if raw is None:
        return IntentConfig()
    if isinstance(raw, IntentConfig):
        return raw
    try:
        if isinstance(raw, dict):
            return IntentConfig.model_validate(raw)
        return IntentConfig.model_validate(raw)
    except Exception:
        return IntentConfig()


class ScoutDataAgent(BaseAgent):
    """Runs live API collection and writes `migration_dataset` to session state."""

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        intent = _parse_intent(ctx.session.state.get("intent_config"))
        # Always collect all data sources - no smart routing
        dataset = await collect_migration_dataset(intent)
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            actions=EventActions(
                state_delta={"migration_dataset": dataset.model_dump(mode="json")}
            ),
        )


class ParallelEdaAgent(BaseAgent):
    """Runs push-factor, destination, pattern, and relocation EDA in parallel."""

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        tracker = get_tracker()
        
        intent = _parse_intent(ctx.session.state.get("intent_config"))
        raw = ctx.session.state.get("migration_dataset") or {}
        try:
            dataset = MigrationDataset.model_validate(raw)
        except Exception as e:
            print(f"[EDA] Failed to validate dataset: {e}")
            dataset = MigrationDataset()
        
        # Debug: Log what data is available
        print(f"\n[EDA DEBUG] Dataset Summary:")
        print(f"  Country: {dataset.country} ({dataset.country_code})")
        print(f"  Intent: {dataset.intent}")
        print(f"  Displacement rows: {len(dataset.displacement or [])}")
        print(f"  Worldbank rows: {len(dataset.worldbank or [])}")
        print(f"  Conflict rows: {len(dataset.conflict_events or [])}")
        print(f"  Climate rows: {len(dataset.climate or [])}")
        print(f"  AQI rows: {len(dataset.aqi or [])}")
        print(f"  News rows: {len(dataset.news or [])}")
        print(f"  Employment rows: {len(dataset.employment or [])}")
        print(f"  City scores rows: {len(dataset.city_scores or [])}")
        print(f"  Destinations: {len(dataset.destinations or [])}")
        print(f"  Missing reasons: {dataset.missing_reasons or []}")
        print()

        # Log EDA start
        tracker.log_step(
            stage="eda_analysis",
            status="started",
            details="Running 4 parallel EDA analyses: Push Factor, Destination, Pattern, Relocation",
            metadata={"analyzers": ["push_factor", "destination", "pattern", "relocation"]},
        )

        def _pf():
            tracker.log_eda_start("Push Factor Analysis")
            result = run_push_factor_analysis(dataset, intent)
            tracker.log_eda_complete("Push Factor Analysis", result.model_dump(mode="json"))
            return result.model_dump(mode="json")

        def _dest():
            tracker.log_eda_start("Destination Tracker")
            result = run_destination_tracker(dataset, intent)
            tracker.log_eda_complete("Destination Tracker", result.model_dump(mode="json"))
            return result.model_dump(mode="json")

        def _pat():
            tracker.log_eda_start("Pattern Detective")
            result = run_pattern_detective(dataset, intent)
            tracker.log_eda_complete("Pattern Detective", result.model_dump(mode="json"))
            return result.model_dump(mode="json")

        def _rel():
            tracker.log_eda_start("Relocation Scorer")
            result = run_relocation_scorer(dataset, intent)
            tracker.log_eda_complete("Relocation Scorer", result.model_dump(mode="json"))
            return result.model_dump(mode="json")

        push_factor_result, destination_result, pattern_result, relocation_result = (
            await asyncio.gather(
                asyncio.to_thread(_pf),
                asyncio.to_thread(_dest),
                asyncio.to_thread(_pat),
                asyncio.to_thread(_rel),
            )
        )
        
        tracker.log_step(
            stage="eda_analysis",
            status="completed",
            details="All EDA analyses finished successfully",
        )

        evidence_snippets = json.dumps(
            {
                "country": dataset.country,
                "country_code": dataset.country_code,
                "target_country": dataset.target_country,
                "target_country_code": dataset.target_country_code,
                "citations": [c.model_dump(mode="json") for c in dataset.citations[:40]],
                "data_freshness": dataset.data_freshness,
                "row_counts": {
                    "worldbank": len(dataset.worldbank),
                    "displacement": len(dataset.displacement),
                    "news": len(dataset.news),
                    "climate": len(dataset.climate),
                    "aqi": len(dataset.aqi),
                },
                "sample_rows": {
                    "destinations": dataset.destinations[:5],
                    "news": dataset.news[:3],
                    "aqi": dataset.aqi[:5],
                    "worldbank": dataset.worldbank[:5],
                },
            },
            default=str,
        )

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            actions=EventActions(
                state_delta={
                    "push_factor_result": push_factor_result,
                    "destination_result": destination_result,
                    "pattern_result": pattern_result,
                    "relocation_result": relocation_result,
                    "evidence_snippets": evidence_snippets,
                }
            ),
        )
