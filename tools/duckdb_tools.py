"""DuckDB cache layer + run_sql_query for EDA agents."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

# TTL per logical table (hours except climate_days)
_TTL_HOURS = {
    "aqi_data": 2,
    "news_articles": 1,
    "conflict_events": 4,
    "economic_indicators": 24,
    "city_scores": 24,
    "climate_data": 24 * 7,
    "displacement_data": 24,
    "employment_data": 24,
}

_DEFAULT_DB = Path(__file__).resolve().parent.parent / "cache" / "migration_intel.duckdb"


def db_path() -> str:
    return os.environ.get("DUCKDB_PATH", str(_DEFAULT_DB))


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS cache_meta (
            table_name VARCHAR,
            cache_key VARCHAR,
            fetched_at TIMESTAMP,
            PRIMARY KEY (table_name, cache_key)
        );
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS economic_indicators (
            country VARCHAR,
            year INTEGER,
            indicator VARCHAR,
            value DOUBLE,
            label VARCHAR,
            endpoint_url VARCHAR,
            fetched_at TIMESTAMP,
            cache_key VARCHAR
        );
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS displacement_data (
            country VARCHAR,
            year INTEGER,
            metric VARCHAR,
            value DOUBLE,
            coa VARCHAR,
            coa_name VARCHAR,
            endpoint_url VARCHAR,
            fetched_at TIMESTAMP,
            cache_key VARCHAR
        );
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS conflict_events (
            country VARCHAR,
            event_date VARCHAR,
            fatalities DOUBLE,
            event_type VARCHAR,
            endpoint_url VARCHAR,
            fetched_at TIMESTAMP,
            cache_key VARCHAR
        );
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS city_scores (
            slug VARCHAR,
            category VARCHAR,
            score_out_of_10 DOUBLE,
            endpoint_url VARCHAR,
            fetched_at TIMESTAMP,
            cache_key VARCHAR
        );
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS news_articles (
            country VARCHAR,
            title VARCHAR,
            source VARCHAR,
            published_at VARCHAR,
            endpoint_url VARCHAR,
            fetched_at TIMESTAMP,
            cache_key VARCHAR
        );
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS climate_data (
            country VARCHAR,
            year INTEGER,
            avg_daily_max_temp_c DOUBLE,
            annual_precipitation_mm DOUBLE,
            endpoint_url VARCHAR,
            fetched_at TIMESTAMP,
            cache_key VARCHAR
        );
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS employment_data (
            country VARCHAR,
            year INTEGER,
            unemployment_rate DOUBLE,
            endpoint_url VARCHAR,
            fetched_at TIMESTAMP,
            cache_key VARCHAR
        );
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS aqi_data (
            country VARCHAR,
            location VARCHAR,
            pm25 DOUBLE,
            endpoint_url VARCHAR,
            fetched_at TIMESTAMP,
            cache_key VARCHAR
        );
        """
    )


def _connect() -> duckdb.DuckDBPyConnection:
    p = db_path()
    _ensure_parent(p)
    con = duckdb.connect(p)
    init_schema(con)
    return con


def cache_key(country: str, year_from: int, year_to: int) -> str:
    return f"{country.upper()}|{year_from}-{year_to}"


_ALLOWED_TABLES = frozenset(
    {
        "economic_indicators",
        "displacement_data",
        "conflict_events",
        "city_scores",
        "news_articles",
        "climate_data",
        "employment_data",
        "aqi_data",
    }
)


def _ttl_ok(con: duckdb.DuckDBPyConnection, table: str, key: str) -> bool:
    hours = _TTL_HOURS.get(table, 24)
    row = con.execute(
        "SELECT fetched_at FROM cache_meta WHERE table_name=? AND cache_key=?",
        [table, key],
    ).fetchone()
    if not row:
        return False
    fetched = row[0]
    if isinstance(fetched, str):
        fetched = datetime.fromisoformat(fetched.replace("Z", "+00:00"))
    if getattr(fetched, "tzinfo", None) is None:
        fetched = fetched.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - fetched < timedelta(hours=hours)


def _touch_meta(con: duckdb.DuckDBPyConnection, table: str, key: str) -> None:
    now = datetime.now(timezone.utc)
    con.execute("DELETE FROM cache_meta WHERE table_name = ? AND cache_key = ?", [table, key])
    con.execute(
        "INSERT INTO cache_meta (table_name, cache_key, fetched_at) VALUES (?, ?, ?)",
        [table, key, now],
    )


def replace_table_rows(
    table: str,
    key: str,
    rows: list[dict[str, Any]],
    columns: list[str],
) -> None:
    """Delete prior rows for cache_key and insert fresh rows."""
    if table not in _ALLOWED_TABLES:
        raise ValueError(f"Unknown table: {table}")
    if not rows:
        return
    con = _connect()
    try:
        con.execute(f"DELETE FROM {table} WHERE cache_key = ?", [key])
        df = pd.DataFrame(rows)
        for col in columns:
            if col not in df.columns:
                df[col] = None
        df = df[columns]
        df["cache_key"] = key
        con.register("_tmp_df", df)
        con.execute(f"INSERT INTO {table} BY NAME SELECT * FROM _tmp_df")
        con.unregister("_tmp_df")
        _touch_meta(con, table, key)
    finally:
        con.close()


def should_use_cache(table: str, key: str) -> bool:
    con = _connect()
    try:
        return _ttl_ok(con, table, key)
    finally:
        con.close()


def run_sql_query(sql: str) -> list[dict[str, Any]]:
    """EDA-facing SQL runner against the local DuckDB cache."""
    lowered = sql.strip().lower()
    if not lowered.startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")
    con = _connect()
    try:
        return con.execute(sql).fetchdf().to_dict(orient="records")
    finally:
        con.close()
