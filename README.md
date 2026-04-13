# Migration Intelligence Agent

Multi-step **Google ADK** pipeline (**Gemini**) that classifies your question, pulls live public data (UNHCR, World Bank, Open-Meteo, OpenAQ, GDELT, optional NewsAPI + ACLED, Teleport), caches it in **DuckDB**, runs parallel **EDA** (correlations, destinations, historical pattern similarity, relocation scoring), and returns a structured **HypothesisReport** with citations. A **Flask** frontend renders workflow steps, EDA outputs, Plotly visuals, citations, and the final hypothesis without raw JSON dumps.

## Setup

1. **Python 3.10+** recommended.

2. Install dependencies with `uv`:

```bash
cd Migration-Intelligence-Agent
uv sync
```

3. Copy `.env.example` to `.env` and add your keys:

```bash
cp .env.example .env
```

Required:

- `GOOGLE_API_KEY` for Gemini / Google ADK

Optional but supported:

- `ACLED_API_KEY` and `ACLED_EMAIL` for conflict events
- `NEWS_API_KEY` for NewsAPI headlines
- `OPENAQ_API_KEY` for higher-rate OpenAQ access
- `OPENMETEO_API_KEY` placeholder for Open-Meteo if your deployment uses a managed key
- `ILOSTAT_API_KEY` placeholder for ILOSTAT if your environment requires one
- `GDELT_API_KEY` placeholder for deployments that proxy or wrap GDELT access

4. Run the web app:

```bash
uv run migration-intel web
```

You can also run it directly:

```bash
uv run python main.py web
uv run python main.py query "Why are people leaving Venezuela?"
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

## Project Requirements Mapping

Required concepts:

- Frontend: [app.py](/Users/shwetatripathi/Migration-Intelligence-Agent/Migration-Intelligence-Agent/app.py) `index()` and `analyze()` render the Flask UI and execute the workflow from the user input box.
- Agent framework: [agents/orchestrator.py](/Users/shwetatripathi/Migration-Intelligence-Agent/Migration-Intelligence-Agent/agents/orchestrator.py) `build_root_agent()` and `run_migration_pipeline()` use Google ADK.
- Tool calling: [agents/scout_service.py](/Users/shwetatripathi/Migration-Intelligence-Agent/Migration-Intelligence-Agent/agents/scout_service.py) `collect_migration_dataset()` calls live data tools; EDA tools are invoked in [agents/push_factor_analyst.py](/Users/shwetatripathi/Migration-Intelligence-Agent/Migration-Intelligence-Agent/agents/push_factor_analyst.py) `run_push_factor_analysis()`, [agents/pattern_detective.py](/Users/shwetatripathi/Migration-Intelligence-Agent/Migration-Intelligence-Agent/agents/pattern_detective.py) `run_pattern_detective()`, and related analyst modules.
- Non-trivial dataset: live external data is collected at runtime in `tools/*.py` and normalized into `MigrationDataset`.
- Multi-agent pattern: [agents/workflow_agents.py](/Users/shwetatripathi/Migration-Intelligence-Agent/Migration-Intelligence-Agent/agents/workflow_agents.py) `ScoutDataAgent` and `ParallelEdaAgent` implement orchestrator plus parallel fan-out EDA.
- Deployed target: the app is run as a Flask service via [main.py](/Users/shwetatripathi/Migration-Intelligence-Agent/Migration-Intelligence-Agent/main.py) `main()` and can be deployed behind any WSGI-compatible host.
- README: this file documents Collect → EDA → Hypothesize and runtime steps.

Grab-bag concepts implemented:

- Code execution: [analysis/correlation.py](/Users/shwetatripathi/Migration-Intelligence-Agent/Migration-Intelligence-Agent/analysis/correlation.py) and [analysis/stats_tools.py](/Users/shwetatripathi/Migration-Intelligence-Agent/Migration-Intelligence-Agent/analysis/stats_tools.py) run pandas/scipy/statsmodels/sklearn computations at runtime.
- Structured output: [models/schemas.py](/Users/shwetatripathi/Migration-Intelligence-Agent/Migration-Intelligence-Agent/models/schemas.py) plus [agents/hypothesis_agent.py](/Users/shwetatripathi/Migration-Intelligence-Agent/Migration-Intelligence-Agent/agents/hypothesis_agent.py) enforce structured agent outputs.
- Data visualization: [analysis/visualization.py](/Users/shwetatripathi/Migration-Intelligence-Agent/Migration-Intelligence-Agent/analysis/visualization.py) `build_migration_panels()` and hypothesis chart panels are rendered in the Flask UI.
- Parallel execution: [agents/workflow_agents.py](/Users/shwetatripathi/Migration-Intelligence-Agent/Migration-Intelligence-Agent/agents/workflow_agents.py) `ParallelEdaAgent._run_async_impl()` executes EDA analysts concurrently.

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
