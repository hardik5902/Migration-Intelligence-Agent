"""Smart tool router: analyzes query to decide which APIs to call."""

from __future__ import annotations

from typing import AsyncGenerator

from typing_extensions import override
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions

from agents.progress_tracker import get_tracker
from models.schemas import ToolSelection, IntentConfig


def _analyze_query_for_tools(raw_message: str, intent: IntentConfig) -> ToolSelection:
    """Smart analysis: route to specific tools based on query keywords and intent."""
    text = (raw_message + " " + intent.intent).lower()
    
    # For relocation_advisory intent, ALWAYS enable all tools (they're all essential)
    if intent.intent == "relocation_advisory":
        return ToolSelection(
            worldbank=True,
            unhcr=True,
            acled=True,
            teleport=True,
            news=True,
            climate=True,
            employment=True,
            aqi=True,
            reasoning="relocation advisory (all tools required)"
        )
    
    selection = ToolSelection()  # Defaults to common tools

    # Count keywords for each tool domain
    conflict_words = sum(1 for word in ["conflict", "war", "violence", "fighting", "attack", "armed", "security", "danger", "terrorist", "rebel"] if word in text)
    economy_words = sum(1 for word in ["economy", "job", "employment", "unemployment", "wage", "income", "gdp", "business", "trade", "economic", "poverty"] if word in text)
    climate_words = sum(1 for word in ["climate", "weather", "heat", "temperature", "drought", "flood", "disaster", "environment", "air", "pollution"] if word in text)
    news_words = sum(1 for word in ["news", "recent", "happening", "current", "today", "latest", "headline", "breaking"] if word in text)
    safety_words = sum(1 for word in ["safe", "safety", "danger", "dangerous", "secure", "threat"] if word in text)
    destination_words = sum(1 for word in ["destination", "where", "move to", "relocate", "go to", "target"] if word in text)

    reasoning_parts = []

    # ACLED: conflict analysis
    if conflict_words > 0 or intent.intent in ["push_factor", "real_time"]:
        selection.acled = True
        reasoning_parts.append(f"conflict analysis ({conflict_words} keywords)")
    
    # NEWS API: real-time or news-focused
    if news_words > 0 or intent.intent == "real_time":
        selection.news = True
        reasoning_parts.append(f"recent news ({news_words} keywords)")
    
    # EMPLOYMENT: job/economy focus
    if economy_words > 2:
        selection.employment = True
        reasoning_parts.append(f"employment analysis ({economy_words} keywords)")
    
    # CLIMATE: climate/environment focus
    if climate_words > 1:
        selection.climate = True
        selection.aqi = True
        reasoning_parts.append(f"climate/environment ({climate_words} keywords)")
    
    # TELEPORT: focus on city scores for destination
    if destination_words > 0:
        selection.teleport = True
        reasoning_parts.append("destination assessment")
    
    # Safety emphasis: ACLED + Teleport
    if safety_words > 0:
        selection.acled = True
        selection.teleport = True
        reasoning_parts.append(f"safety focus ({safety_words} keywords)")
    
    # Destination intent: need UNHCR data
    if intent.intent == "destination":
        selection.unhcr = True
        selection.worldbank = True
        selection.teleport = True
        reasoning_parts.append("destination tracking (UNHCR + economic)")

    selection.reasoning = "; ".join(reasoning_parts) if reasoning_parts else "default tool selection"
    return selection


class ToolRouterAgent(BaseAgent):
    """Analyzes query to decide which APIs are most relevant."""

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        # Get intent from session
        intent_data = ctx.session.state.get("intent_config") or {}
        try:
            intent = IntentConfig.model_validate(intent_data)
        except Exception:
            intent = IntentConfig()

        # Get raw user message (search through conversation)
        raw_message = ""
        
        # Try to extract from message history
        if hasattr(ctx, "messages") and ctx.messages:
            for msg in reversed(ctx.messages):
                if hasattr(msg, "content"):
                    content = msg.content
                    if hasattr(content, "parts"):
                        for part in content.parts:
                            if hasattr(part, "text"):
                                raw_message = part.text
                                break
                    elif isinstance(content, str):
                        raw_message = content
                        break
        
        # Perform smart analysis
        tool_selection = _analyze_query_for_tools(raw_message, intent)
        
        # Log tool selection
        tracker = get_tracker()
        tracker.log_step(
            stage="tool_routing",
            status="completed",
            details=f"Intent: {intent.intent} | {tool_selection.reasoning}",
            metadata={"intent": intent.intent, "tool_selection": tool_selection.model_dump()},
        )
        
        # Log which tools are enabled
        for tool_name, enabled in [
            ("World Bank", tool_selection.worldbank),
            ("UNHCR", tool_selection.unhcr),
            ("ACLED", tool_selection.acled),
            ("Teleport", tool_selection.teleport),
            ("NewsAPI", tool_selection.news),
            ("Climate", tool_selection.climate),
            ("Employment", tool_selection.employment),
            ("AQI", tool_selection.aqi),
        ]:
            if enabled:
                tracker.log_tool_selection(tool_name, enabled, "selected")

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            actions=EventActions(
                state_delta={"tool_selection": tool_selection.model_dump()}
            ),
        )
