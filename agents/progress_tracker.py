"""Real-time progress tracking for the pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable


class PipelineTracker:
    """Tracks and logs pipeline execution for transparency."""

    def __init__(self):
        self.events: list[dict[str, Any]] = []
        self.callbacks: list[Callable] = []

    def add_callback(self, func: Callable) -> None:
        """Register a callback to receive progress updates."""
        self.callbacks.append(func)

    def log_step(
        self,
        stage: str,
        status: str,
        details: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log a pipeline step with timestamp."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "status": status,
            "details": details,
            "metadata": metadata or {},
        }
        self.events.append(event)

        # Notify all callbacks
        for callback in self.callbacks:
            try:
                callback(event)
            except Exception:
                pass

    def log_tool_selection(self, tool_name: str, enabled: bool, reason: str = "") -> None:
        """Log which tools are selected."""
        self.log_step(
            stage="tool_routing",
            status="enabled" if enabled else "disabled",
            details=f"{tool_name}: {reason}" if reason else tool_name,
            metadata={"tool": tool_name, "enabled": enabled},
        )

    def log_tool_call(
        self,
        tool_name: str,
        params: dict[str, Any],
        status: str = "calling",
        result_count: int | None = None,
        error: str | None = None,
    ) -> None:
        """Log an API tool call."""
        self.log_step(
            stage="data_collection",
            status=status,
            details=f"{tool_name}({', '.join(f'{k}={v}' for k, v in list(params.items())[:2])})",
            metadata={
                "tool": tool_name,
                "params": params,
                "result_count": result_count,
                "error": error,
            },
        )

    def log_eda_start(self, analyzer_name: str) -> None:
        """Log start of EDA analysis."""
        self.log_step(
            stage="eda_analysis",
            status="started",
            details=f"Running {analyzer_name}",
            metadata={"analyzer": analyzer_name},
        )

    def log_eda_complete(
        self, analyzer_name: str, results: dict[str, Any] | None = None
    ) -> None:
        """Log completion of EDA analysis."""
        summary = ""
        if results:
            if isinstance(results, dict):
                if "top_driver" in results:
                    summary = f"Top driver: {results.get('top_driver')}"
                elif "top_destinations" in results:
                    summary = f"Found {len(results.get('top_destinations', []))} destinations"
                elif "top_countries" in results:
                    summary = f"Scored {len(results.get('top_countries', []))} countries"
                elif "template" in results:
                    summary = f"Pattern: {results.get('template')}"

        self.log_step(
            stage="eda_analysis",
            status="completed",
            details=f"{analyzer_name} finished. {summary}",
            metadata={"analyzer": analyzer_name, "summary": summary},
        )

    def get_summary(self) -> str:
        """Return a text summary of all events."""
        lines = ["=== Pipeline Execution Summary ===", ""]
        stages = {}
        for event in self.events:
            stage = event["stage"]
            if stage not in stages:
                stages[stage] = []
            stages[stage].append(event)

        for stage, events in stages.items():
            lines.append(f"📊 {stage.upper()}")
            for event in events:
                status_icon = {
                    "started": "▶️",
                    "calling": "🔄",
                    "completed": "✅",
                    "enabled": "✓",
                    "disabled": "✗",
                }.get(event["status"], "•")
                lines.append(f"  {status_icon} {event['details']}")
            lines.append("")

        return "\n".join(lines)

    def get_json(self) -> str:
        """Return all events as JSON."""
        return json.dumps(self.events, indent=2)


# Global tracker instance
_global_tracker: PipelineTracker | None = None


def get_tracker() -> PipelineTracker:
    """Get or create the global tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = PipelineTracker()
    return _global_tracker


def reset_tracker() -> None:
    """Reset the global tracker."""
    global _global_tracker
    _global_tracker = PipelineTracker()
