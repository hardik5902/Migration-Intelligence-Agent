"""
Google ADK root agent for `adk run` / `adk web`.

From this repository root (with `.env` containing GOOGLE_API_KEY):

  adk web

Select the app that corresponds to this folder, or run per ADK CLI docs for your version.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from agents.orchestrator import build_root_agent

root_agent = build_root_agent()
