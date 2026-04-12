"""Scout / data collection step.

The live implementation is `ScoutDataAgent` in `agents.workflow_agents`, which
calls `agents.scout_service.collect_migration_dataset` and writes
`migration_dataset` into ADK session state.
"""

from __future__ import annotations

from agents.workflow_agents import ScoutDataAgent

__all__ = ["ScoutDataAgent"]
