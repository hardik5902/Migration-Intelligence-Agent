"""EDA helpers exposed as plain callables (wrap DuckDB + stats for tools / tests).

Use with `google.adk.tools.function_tool.FunctionTool` if you attach them to an LlmAgent.
"""

from __future__ import annotations

from analysis.correlation import (
    run_correlation_analysis,
    run_granger_test,
    run_growth_rate,
    run_lag_analysis,
)
from analysis.stats_tools import (
    haversine_distance,
    run_anomaly_detect,
    run_cosine_similarity,
)
from tools.duckdb_tools import run_sql_query
from tools.news_tools import gdelt_sentiment_score

__all__ = [
    "run_sql_query",
    "run_correlation_analysis",
    "run_granger_test",
    "run_lag_analysis",
    "run_growth_rate",
    "run_cosine_similarity",
    "haversine_distance",
    "run_anomaly_detect",
    "gdelt_sentiment_score",
]
