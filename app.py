"""Streamlit UI for the Migration Intelligence Agent (Google ADK + Gemini)."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import plotly.io as pio
import streamlit as st
from dotenv import load_dotenv

from agents.orchestrator import run_migration_pipeline
from analysis.visualization import build_migration_panels
from models.schemas import HypothesisReport

load_dotenv(Path(__file__).resolve().parent / ".env")

st.set_page_config(page_title="Migration Intelligence Agent", layout="wide")
st.title("Migration Intelligence Agent")
st.caption("Google ADK · Gemini · live UNHCR / World Bank / Open-Meteo / OpenAQ / News / GDELT (+ optional ACLED, NewsAPI)")

with st.sidebar:
    st.subheader("Controls")
    year_from = st.slider("Year from", 2000, 2024, 2010)
    year_to = st.slider("Year to", 2000, 2025, 2023)
    override = st.text_input("Intent override (optional)", "")
    aqi_w = st.slider("Relocation weight: AQI", 0.0, 1.0, 0.4)
    health_w = st.slider("Relocation weight: healthcare", 0.0, 1.0, 0.35)
    edu_w = st.slider("Relocation weight: education", 0.0, 1.0, 0.25)

query = st.text_area(
    "Your question",
    value="Why are people leaving Venezuela?",
    height=100,
)

if st.button("Run pipeline", type="primary"):
    if not os.environ.get("GOOGLE_API_KEY"):
        st.error("Set GOOGLE_API_KEY in a `.env` file (see `.env.example`).")
    else:
        hint = query.strip()
        hint = (
            f"{hint}\n\n[Context: analysis years {year_from}-{year_to}. "
            f"Relocation weights if applicable — aqi:{aqi_w}, healthcare:{health_w}, education:{edu_w}]"
        )
        if override.strip():
            hint = f"{hint}\n\n(User intent override: {override.strip()})"
        with st.spinner("Collecting data…"):
            pass
        with st.spinner("Running ADK pipeline (intent → scout → EDA → hypothesis)…"):
            try:

                async def _run():
                    return await run_migration_pipeline(
                        hint,
                        year_from=year_from,
                        year_to=year_to,
                    )

                state = asyncio.run(_run())
            except Exception as exc:
                st.exception(exc)
                state = {}

        md_raw = state.get("migration_dataset")
        if isinstance(md_raw, dict):
            try:
                fig_json = build_migration_panels(md_raw)
                fig = pio.from_json(fig_json)
                st.subheader("Exploratory panels")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.warning(f"Chart build skipped: {exc}")

        hyp_raw = state.get("hypothesis_report")
        if hyp_raw:
            try:
                hyp = HypothesisReport.model_validate(hyp_raw)
            except Exception:
                hyp = None
            if hyp:
                st.subheader("Headline")
                st.write(hyp.headline)
                st.metric("Confidence", f"{hyp.confidence_score:.2f}")
                if hyp.supporting_points:
                    st.subheader("Supporting points")
                    st.dataframe(
                        [p.model_dump() for p in hyp.supporting_points],
                        use_container_width=True,
                    )
                if hyp.citations:
                    st.subheader("Citations")
                    st.dataframe(
                        [c.model_dump() for c in hyp.citations],
                        use_container_width=True,
                    )
                if hyp.competing_hypotheses:
                    st.subheader("Competing hypotheses")
                    for ch in hyp.competing_hypotheses:
                        with st.expander(ch.hypothesis):
                            st.write("For:", ch.evidence_for)
                            st.write("Against:", ch.evidence_against)
                            st.caption(f"p={ch.probability_score:.2f}")
                if hyp.recent_headlines:
                    st.subheader("Headlines")
                    for n in hyp.recent_headlines[:5]:
                        st.markdown(f"**{n.title}** — _{n.source}_ ({n.sentiment_score:.2f})")
                if hyp.charts:
                    for panel in hyp.charts:
                        try:
                            st.plotly_chart(pio.from_json(panel.fig_json), use_container_width=True)
                            st.caption(panel.caption)
                        except Exception:
                            pass
                with st.expander("Full HypothesisReport JSON"):
                    st.json(json.loads(hyp.model_dump_json()))
        else:
            st.info("No hypothesis_report in session state (pipeline may have failed early).")

        with st.expander("Session state keys"):
            st.json({k: type(v).__name__ for k, v in state.items()})

st.markdown(
    "---\nWeights from sidebar are passed into the query string when you use "
    "**relocation** phrasing so the intent agent can pick them up."
)
