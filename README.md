# Migration Intelligence Agent

Multi-step **Google ADK** pipeline (**Gemini**) that classifies your question, pulls live public data (UNHCR, World Bank, Open-Meteo, OpenAQ, GDELT, optional NewsAPI + ACLED, Teleport), caches it in **DuckDB**, runs parallel **EDA** (correlations, destinations, historical pattern similarity, relocation scoring), and returns a structured **HypothesisReport** with citations. A **Streamlit** UI renders tables, Plotly panels, and the final JSON.

## Setup

1. **Python 3.10+** recommended.

2. Create a virtual environment and install dependencies:

```bash
cd Migration-Intelligence-Agent
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and fill in keys:

- **`GOOGLE_API_KEY`** (required) — [Google AI Studio](https://aistudio.google.com/app/apikey) for Gemini / ADK.
- **`ACLED_API_KEY`** + **`ACLED_EMAIL`** (optional) — [ACLED](https://developer.acleddata.com/) for conflict events.
- **`NEWS_API_KEY`** (optional) — [NewsAPI](https://newsapi.org/) for headlines (GDELT works without a key).

4. Run the UI:

```bash
streamlit run app.py
```

## CLI smoke test (optional)

```bash
python3 scripts/run_once.py "Why are people leaving Venezuela?" --year-from 2015 --year-to 2022
```

## ADK dev UI (optional)

Repository root includes `agent.py` exporting `root_agent`. From the same directory:

```bash
adk web
```

…per [ADK Python quickstart](https://google.github.io/adk-docs/get-started/python/). Agents are built in `agents/orchestrator.py` (`build_root_agent`).

## Architecture (short)

1. **`intent_classifier`** (`LlmAgent` / `Agent`) — structured `IntentConfig` (`output_schema`).
2. **`scout_data_agent`** (`BaseAgent`) — async HTTP + DuckDB TTL cache → `migration_dataset`.
3. **`parallel_eda_agent`** (`BaseAgent`) — asyncio parallel Python EDA → `push_factor_result`, `destination_result`, `pattern_result`, `relocation_result`, `evidence_snippets`.
4. **`hypothesis_agent`** (`LlmAgent`) — structured `HypothesisReport` (`output_schema`), no tools (ADK constraint).

## Example queries

- Push factors: *“Why are people leaving Venezuela?”*
- Destinations: *“Where are Syrians going?”*
- Historical: *“Is Sudan comparable to Zimbabwe in 2007?”*
- Real-time: *“What’s happening now in Sudan?”*
- Relocation: *“I live in India; I want lower PM2.5 and better education — where should I look?”*

## Data & cache

DuckDB file default: `cache/migration_intel.duckdb` (override with `DUCKDB_PATH`). TTLs follow the build plan (e.g. news ~1h, ACLED ~4h, climate ~7d).

## License

See `LICENSE`.
