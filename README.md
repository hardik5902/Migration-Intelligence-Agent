# Migration Intelligence Agent

**Live demo:** https://migration-intelligence-agent-1008110254578.us-central1.run.app/

A **multi-agent data analysis system** built on **Google ADK + Gemini** that performs a complete analyst workflow, collects live data, explores it statistically, and forms evidence-backed hypotheses.

---

## What it does

Ask a question like *"Which countries have the best quality of life for migrants?"* and the system:

1. **Picks the right data tools and countries** — an LLM reads your query, selects up to 5 relevant APIs and K best-fit countries
2. **Fetches live data in parallel** — up to 6 APIs fetch simultaneously for every country
3. **Runs exploratory data analysis** — CAGR, anomaly detection, Pearson/Spearman correlations, and statistical summaries in pure Python
4. **Builds 4 comparison charts + 4 EDA charts** — Plotly visuals selected by a second LLM call grounded in EDA findings
5. **Delivers 3 data-grounded hypotheses** — each cites specific numbers, tests a competing explanation, and carries a confidence score


---
---


## Example queries

| Category | Query |
|----------|-------|
| Quality of life | *"Which 5 countries have the best quality of life for migrants?"* |
| Economic | *"Compare GDP growth and inflation in Germany, Canada and Australia"* |
| Healthcare | *"Top 3 countries with highest healthcare spending and best life expectancy"* |
| Environment | *"Countries with the cleanest air and lowest CO2 emissions"* |
| Safety | *"Safest countries with lowest conflict events for relocation"* |
| Infrastructure | *"Best countries for internet access and electricity infrastructure"* |
| Labour | *"Where is youth unemployment lowest in Europe?"* |

---

## The Three Steps

### Step 1 — Collect


The `DataCollectorADKAgent`  runs **K countries in parallel**. For each country, fans out across **up to 6 live external APIs simultaneously**:

| Tool | Source | Data |
|------|--------|------|
| `worldbank_tools.fetch_macro_bundle()` | World Bank REST API | 25 indicators — GDP, inflation, health, education, governance, infrastructure |
| `employment_tools.get_employment_data()` | ILOSTAT API | Unemployment rate, youth unemployment, labour force participation |
| `environment_tools.get_environment_data()` | Open-Meteo Archive + OpenAQ / World Bank PM2.5 | Temperature anomaly, precipitation, PM2.5 air quality |
| `acled_tools.get_conflict_events()` | ACLED API | Armed conflict events, fatalities, event types |
| `teleport_tools.get_city_scores()` | Teleport API (World Bank composite) | Quality-of-life scores across 17 categories |
| `news_tools.get_country_news()` + `get_gdelt_sentiment()` | NewsAPI + GDELT | Recent headlines + sentiment score |

All fetched data is stored in a **local DuckDB file** (`cache/migration_intel.duckdb`) with per-table TTLs (1h news → 7d climate). 
On the next query for the same country/period, the cache is read via **SQL** instead of hitting the API again.

The LLM first determines which tools are relevant, so a healthcare query does not trigger ACLED and a conflict query does not trigger Teleport.

### Step 2 — Explore and Analyse (EDA)

The EDA agent  runs four statistical passes over the collected data:

| Method | File / Function | What it surfaces |
|--------|----------------|-----------------|
| **CAGR** | `analysis/correlation.py` → `run_growth_rate()` | Compound annual growth rate + peak/trough year per metric per country |
| **Anomaly detection** | `analysis/stats_tools.py` → `run_anomaly_detect()` | Z-score \|z\| > 2 outlier years per time series |
| **Cross-country correlation** | `analysis/correlation.py` → `run_correlation_analysis()` | Pearson + Spearman r, p-value, significance flag per indicator pair |
| **Statistical summary** | `agents/eda_analyst.py` → `run_eda()` | Mean ± σ, min/max, n per metric per country |

Results are stored in `eda_findings` (session state) and emitted as finding cards to the browser. Four EDA-specific Plotly charts are built (`analysis/eda_charts.py`): correlation heatmap, CAGR bar chart, anomaly timeline, distribution box plot.

The EDA is **query-adaptive**: a safety query drives analysis of conflict events and political stability; an economic query drives GDP/inflation CAGR analysis. Metrics are selected dynamically from `selected_tools` — the agent does not compute irrelevant statistics.

### Step 3 — Hypothesise

**Where:** `agents/adk_agents.py` → `UnifiedAnalysisADKAgent._run_async_impl()`, calling `agents/analysis_agent.py` → `run_unified_analysis()`

A single LLM call (Gemini) receives the **complete EDA output** — statistical findings, data coverage per country, chart manifests — and returns a structured JSON object with:

- **`comparison_chart_keys`** — 4 chart keys from the pre-built comparison registry
- **`eda_chart_keys`** — up to 4 statistical chart keys (no metric overlap with comparison charts)
- **`hypotheses`** — exactly 3 `HypothesisInsight` objects, each with:
  - `headline` + `summary` citing a specific number
  - `evidence_for` list in *"Country/metric: value (year)"* format
  - `evidence_against` caveats
  - `competing_hypothesis` + `competing_verdict`
  - `confidence` score (85-100 = multi-year trend; <45 = sparse data)

The hypotheses are **grounded in EDA output, not model weights**. The system instruction explicitly forbids generic claims and requires every hypothesis to cite a specific data point from the EDA findings passed as context.

---

## Architecture

```
Browser (SSE)
    │
    ▼
Flask / app.py  ──────────────────────────────────────────────────────
    │  POST /analyze                                                   │
    ▼                                                                  │
Google ADK SequentialAgent (agents/adk_agents.py)                     │
    │                                                                  │
    ├─► ToolSelectorADKAgent          (LLM — Gemini)                  │
    │     agents/tool_selector.py                                      │
    │     • Classifies query scope                                     │
    │     • Selects tools + K countries                                │
    │     • Emits SSE: stage, selection                                │
    │                                                                  │
    ├─► DataCollectorADKAgent         (async Python)                   │
    │     agents/data_collector.py + agents/scout_service.py           │
    │     • asyncio.gather: K countries in parallel                    │
    │     • Per-country: asyncio.gather across 6 APIs                  │
    │     • DuckDB TTL cache read/write                                │
    │     • Emits SSE: stage, selection (with coverage ranking)        │
    │                                                                  │
    ├─► EDAAnalystADKAgent            (pure Python — no LLM)           │
    │     agents/eda_analyst.py                                        │
    │     analysis/correlation.py, stats_tools.py, eda_charts.py      │
    │     • CAGR, anomaly z-score, Pearson/Spearman correlations       │
    │     • Builds comparison + EDA chart manifests                    │
    │     • Emits SSE: stage, eda (finding cards)                      │
    │                                                                  │
    └─► UnifiedAnalysisADKAgent       (LLM — Gemini)                   │
          agents/analysis_agent.py                                     │
          analysis/country_charts.py, eda_charts.py                   │
          • Single LLM call: chart keys + 3 hypotheses                 │
          • Renders Plotly charts (comparison + EDA)                   │
          • Emits SSE: eda_charts, charts, evidence, stage, done       │
                                                                       │
Tools (all in tools/)                                                  │
  worldbank_tools.py  →  World Bank REST API (25 indicators)           │
  employment_tools.py →  ILOSTAT API                                   │
  environment_tools.py→  Open-Meteo + OpenAQ / WB PM2.5               │
  acled_tools.py      →  ACLED conflict API                            │
  teleport_tools.py   →  Teleport quality-of-life API                  │
  news_tools.py       →  NewsAPI + GDELT                               │
  duckdb_tools.py     →  DuckDB cache (SQL read/write)                 │
```

---

## Setup

### Prerequisites

- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) package manager

### Install

```bash
cd Migration-Intelligence-Agent
uv sync
```

### Configure

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Required
GOOGLE_API_KEY=your_gemini_api_key

# Optional — enables additional data sources
ACLED_API_KEY=your_key
ACLED_EMAIL=your@email.com
NEWS_API_KEY=your_newsapi_key
OPENAQ_API_KEY=your_openaq_key
```

> Without optional keys the system still works: ACLED returns empty (no conflict data), NewsAPI is skipped, OpenAQ falls back to World Bank PM2.5 satellite data.

### Run

```bash
uv run migration-intel web
# or
uv run python main.py web
```

Open **http://localhost:8080**

### ADK dev UI (optional)

```bash
adk web   # from the project root, uses agent.py → root_agent
```

### CLI mode

```bash
uv run python main.py query "Compare healthcare in Germany, Japan and Canada"
```



## Data & cache

- **DuckDB file:** `cache/migration_intel.duckdb` (override with `DUCKDB_PATH` env var)
- **Cache viewer:** visit `/cache` in the browser to inspect cached tables, row counts, and TTL status
- **Cache clear:** POST to `/cache/clear` or click the button in the cache viewer

| Table | TTL | Source |
|-------|-----|--------|
| `economic_indicators` | 24h | World Bank API |
| `employment_data` | 24h | ILOSTAT |
| `city_scores` | 24h | Teleport API |
| `climate_data` | 7 days | Open-Meteo Archive |
| `aqi_data` | 2h | OpenAQ + World Bank PM2.5 |
| `conflict_events` | 4h | ACLED API |
| `news_articles` | 1h | NewsAPI |


## License

See `LICENSE`.
