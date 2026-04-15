"""Unified local entrypoint for the Migration Intelligence project."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def _check_env() -> int:
    keys = [
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "ACLED_USERNAME",
        "ACLED_PASSWORD",
        "NEWS_API_KEY",
    ]
    print(f"Project root: {ROOT}")
    print(f".env file: {'FOUND' if (ROOT / '.env').exists() else 'MISSING'}")
    
    # Check Vertex AI status
    has_vertex_ai = (
        os.environ.get("GOOGLE_GENAI_USE_VERTEXAI") == "TRUE"
        and os.environ.get("GOOGLE_CLOUD_PROJECT")
    )
    print(f"Vertex AI configured: {'✓ YES' if has_vertex_ai else '✗ NO'}")
    
    for key in keys:
        status = "SET" if os.environ.get(key) else "MISSING"
        print(f"  {key}: {status}")
    return 0


async def _run_query(query: str) -> int:
    from agents.orchestrator import run_migration_pipeline
    from agents.debug_inspector import print_pipeline_status
    from agents.progress_tracker import reset_tracker

    # Check for either Vertex AI or direct API key authentication
    has_vertex_ai = (
        os.environ.get("GOOGLE_GENAI_USE_VERTEXAI") == "TRUE"
        and os.environ.get("GOOGLE_CLOUD_PROJECT")
    )
    has_api_key = bool(os.environ.get("GOOGLE_API_KEY"))
    has_app_creds = bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
    
    if not (has_vertex_ai or has_api_key or has_app_creds):
        print(
            "Missing Google authentication. Configure one of:",
            file=sys.stderr,
        )
        print(
            "  1. Vertex AI: GOOGLE_CLOUD_PROJECT + GOOGLE_GENAI_USE_VERTEXAI=TRUE",
            file=sys.stderr,
        )
        print(
            "  2. API Key: GOOGLE_API_KEY",
            file=sys.stderr,
        )
        print(
            "  3. Service Account: GOOGLE_APPLICATION_CREDENTIALS",
            file=sys.stderr,
        )
        return 2

    # Reset tracker for fresh run
    reset_tracker()
    
    print(f"\n🚀 Starting pipeline for query: '{query}'\n")
    
    state = await run_migration_pipeline(query)
    print("Pipeline completed.")
    
    # Print execution status
    print_pipeline_status()
    
    print("State keys:", ", ".join(sorted(state.keys())))
    hypothesis = state.get("hypothesis_report")
    if hypothesis and isinstance(hypothesis, dict):
        print(hypothesis.get("headline", ""))
    return 0


def _run_web(host: str, port: int, debug: bool) -> int:
    from app import app

    app.run(host=host, port=port, debug=debug)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Migration Intelligence project.")
    subparsers = parser.add_subparsers(dest="command")

    env_parser = subparsers.add_parser("check-env", help="Show environment readiness.")
    env_parser.set_defaults(handler=lambda args: _check_env())

    query_parser = subparsers.add_parser("query", help="Run the full pipeline once.")
    query_parser.add_argument("query", nargs="?", default="Why are people leaving Venezuela?")
    query_parser.set_defaults(handler=lambda args: asyncio.run(_run_query(args.query)))

    web_parser = subparsers.add_parser("web", help="Start the Flask frontend.")
    web_parser.add_argument("--host", default="127.0.0.1")
    web_parser.add_argument("--port", type=int, default=8000)
    web_parser.add_argument("--debug", action="store_true")
    web_parser.set_defaults(handler=lambda args: _run_web(args.host, args.port, args.debug))

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.command:
        args = parser.parse_args(["web"])
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
