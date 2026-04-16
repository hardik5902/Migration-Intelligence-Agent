"""Final structured report agent (Gemini + HypothesisReport schema)."""

from __future__ import annotations

import os

from google.adk.agents.llm_agent import Agent

from agents.prompts import HYPOTHESIS_INSTRUCTION
from models.schemas import HypothesisReport

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")


def build_hypothesis_agent() -> Agent:
    return Agent(
        name="hypothesis_agent",
        model=GEMINI_MODEL,
        instruction=HYPOTHESIS_INSTRUCTION,
        output_schema=HypothesisReport,
        output_key="hypothesis_report",
        description="Produces structured HypothesisReport with citations.",
    )
