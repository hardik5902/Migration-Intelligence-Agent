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
        intent = _parse_intent(ctx.session.state.get("intent_config"))
        raw = ctx.session.state.get("migration_dataset") or {}
        try:
            dataset = MigrationDataset.model_validate(raw)
        except Exception:
            dataset = MigrationDataset()

        def _pf():
            return run_push_factor_analysis(dataset, intent).model_dump(mode="json")

        def _dest():
            return run_destination_tracker(dataset, intent).model_dump(mode="json")

        def _pat():
            return run_pattern_detective(dataset, intent).model_dump(mode="json")

        def _rel():
            return run_relocation_scorer(dataset, intent).model_dump(mode="json")

        push_factor_result, destination_result, pattern_result, relocation_result = (
            await asyncio.gather(
                asyncio.to_thread(_pf),
                asyncio.to_thread(_dest),
                asyncio.to_thread(_pat),
                asyncio.to_thread(_rel),
            )
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
