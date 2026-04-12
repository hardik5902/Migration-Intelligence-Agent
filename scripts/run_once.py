#!/usr/bin/env python3
"""One-shot CLI test: requires GOOGLE_API_KEY and network."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from agents.orchestrator import run_migration_pipeline  # noqa: E402


async def main() -> None:
    p = argparse.ArgumentParser(description="Run migration pipeline once")
    p.add_argument("query", nargs="?", default="Why are people leaving Venezuela?")
    p.add_argument("--year-from", type=int, default=2010)
    p.add_argument("--year-to", type=int, default=2023)
    args = p.parse_args()

    state = await run_migration_pipeline(
        args.query,
        year_from=args.year_from,
        year_to=args.year_to,
    )
    hyp = state.get("hypothesis_report")
    if hyp:
        print(json.dumps(hyp, indent=2, default=str)[:8000])
    else:
        print("No hypothesis_report; state keys:", list(state.keys()))


if __name__ == "__main__":
    asyncio.run(main())
