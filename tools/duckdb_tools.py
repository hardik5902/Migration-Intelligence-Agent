"""Cache backend abstraction with local DuckDB and optional GCS storage."""

from __future__ import annotations

import json
import os
import re
import threading as _threading
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import duckdb
import pandas as pd
from google.cloud import storage

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

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DB = (
    Path("/tmp/migration_intel/cache/migration_intel.duckdb")
    if os.environ.get("K_SERVICE")
    else _ROOT / "cache" / "migration_intel.duckdb"
)
_SELECT_BY_KEY_RE = re.compile(
    r"^select \* from (?P<table>[a-z_]+) where cache_key = '(?P<key>[^']+)'$",
    re.IGNORECASE,
)
_storage_client: storage.Client | None = None
_thread_local = _threading.local()


def active_cache_backend() -> str:
    configured = (os.environ.get("CACHE_BACKEND") or "").strip().lower()
    if configured in {"duckdb", "gcs"}:
        return configured
    return "gcs" if _cache_bucket() else "duckdb"


def _cache_bucket() -> str:
    return (os.environ.get("CACHE_BUCKET") or "").strip()


def _cache_prefix() -> str:
    return (os.environ.get("CACHE_PREFIX") or "migration-intel-cache").strip("/")


def db_path() -> str:
    if active_cache_backend() == "gcs":
        return f"gs://{_cache_bucket()}/{_cache_prefix()}"
    return os.environ.get("DUCKDB_PATH", str(_DEFAULT_DB))


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


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
            country VARCHAR,
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
            avg_temp_anomaly_c DOUBLE,
            extreme_heat_days DOUBLE,
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
            youth_unemployment_rate DOUBLE,
            labor_force_participation DOUBLE,
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
            city VARCHAR,
            pm25 DOUBLE,
            endpoint_url VARCHAR,
            fetched_at TIMESTAMP,
            cache_key VARCHAR
        );
        """
    )
    con.execute("ALTER TABLE city_scores ADD COLUMN IF NOT EXISTS country VARCHAR;")
    con.execute("ALTER TABLE climate_data ADD COLUMN IF NOT EXISTS avg_temp_anomaly_c DOUBLE;")
    con.execute("ALTER TABLE climate_data ADD COLUMN IF NOT EXISTS extreme_heat_days DOUBLE;")
    con.execute("ALTER TABLE employment_data ADD COLUMN IF NOT EXISTS youth_unemployment_rate DOUBLE;")
    con.execute("ALTER TABLE employment_data ADD COLUMN IF NOT EXISTS labor_force_participation DOUBLE;")
    con.execute("ALTER TABLE aqi_data ADD COLUMN IF NOT EXISTS city VARCHAR;")


def _connect() -> duckdb.DuckDBPyConnection:
    """Return a per-thread cached DuckDB connection for local cache mode."""
    if active_cache_backend() != "duckdb":
        raise RuntimeError("DuckDB connection requested while cache backend is not duckdb.")

    con = getattr(_thread_local, "con", None)
    if con is None:
        p = db_path()
        _ensure_parent(p)
        con = duckdb.connect(p)
        init_schema(con)
        _thread_local.con = con
    return con


def _get_storage_client() -> storage.Client:
    global _storage_client
    if _storage_client is None:
        _storage_client = storage.Client()
    return _storage_client


def _gcs_blob_name(table: str, key: str) -> str:
    safe_key = quote(key, safe="")
    return f"{_cache_prefix()}/{table}/{safe_key}.json"


def _gcs_read_entry(table: str, key: str) -> dict[str, Any] | None:
    blob = _get_storage_client().bucket(_cache_bucket()).blob(_gcs_blob_name(table, key))
    if not blob.exists():
        return None
    return json.loads(blob.download_as_text())


def _gcs_write_entry(table: str, key: str, rows: list[dict[str, Any]]) -> None:
    entry = {
        "table_name": table,
        "cache_key": key,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
    }
    blob = _get_storage_client().bucket(_cache_bucket()).blob(_gcs_blob_name(table, key))
    blob.upload_from_string(
        json.dumps(entry, default=_json_default),
        content_type="application/json",
    )


def _gcs_iter_entries(table: str) -> list[dict[str, Any]]:
    bucket = _get_storage_client().bucket(_cache_bucket())
    prefix = f"{_cache_prefix()}/{table}/"
    entries: list[dict[str, Any]] = []
    for blob in bucket.list_blobs(prefix=prefix):
        try:
            entries.append(json.loads(blob.download_as_text()))
        except Exception:
            continue
    return entries


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except Exception:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


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


def _ttl_ok(table: str, key: str) -> bool:
    hours = _TTL_HOURS.get(table, 24)
    if active_cache_backend() == "gcs":
        entry = _gcs_read_entry(table, key)
        fetched = _parse_timestamp(entry.get("fetched_at") if entry else None)
        if not fetched:
            return False
        return datetime.now(timezone.utc) - fetched < timedelta(hours=hours)

    row = _connect().execute(
        "SELECT fetched_at FROM cache_meta WHERE table_name=? AND cache_key=?",
        [table, key],
    ).fetchone()
    if not row:
        return False
    fetched = _parse_timestamp(row[0])
    if not fetched:
        return False
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

    if active_cache_backend() == "gcs":
        normalized_rows: list[dict[str, Any]] = []
        for row in rows:
            normalized_row = {col: row.get(col) for col in columns}
            normalized_row["cache_key"] = key
            normalized_rows.append(normalized_row)
        _gcs_write_entry(table, key, normalized_rows)
        return

    con = _connect()
    if not rows:
        con.execute(f"DELETE FROM {table} WHERE cache_key = ?", [key])
        _touch_meta(con, table, key)
        return

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


def should_use_cache(table: str, key: str) -> bool:
    return _ttl_ok(table, key)


def run_sql_query(sql: str) -> list[dict[str, Any]]:
    """Cache-facing SELECT runner.

    For GCS cache mode, only ``SELECT * FROM <table> WHERE cache_key = '...'`` is
    supported, which matches the runtime query pattern in the data collector.
    """
    stripped = sql.strip()
    if not stripped.lower().startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")

    if active_cache_backend() == "gcs":
        match = _SELECT_BY_KEY_RE.match(stripped)
        if not match:
            raise ValueError("GCS cache mode only supports SELECT * ... WHERE cache_key queries.")
        entry = _gcs_read_entry(match.group("table"), match.group("key"))
        return list(entry.get("rows", [])) if entry else []

    return _connect().execute(sql).fetchdf().to_dict(orient="records")


def _annotate_meta(table: str, meta: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ttl_hours = _TTL_HOURS.get(table, 24)
    now = datetime.now(timezone.utc)
    annotated: list[dict[str, Any]] = []
    for item in meta:
        fetched = _parse_timestamp(item.get("fetched_at"))
        annotated_item = dict(item)
        if fetched:
            age = now - fetched
            remaining = timedelta(hours=ttl_hours) - age
            annotated_item["age_min"] = int(age.total_seconds() / 60)
            annotated_item["valid"] = remaining.total_seconds() > 0
            annotated_item["expires_in"] = (
                f"{int(remaining.total_seconds() // 3600)}h "
                f"{int((remaining.total_seconds() % 3600) // 60)}m"
                if annotated_item["valid"]
                else "expired"
            )
        else:
            annotated_item["age_min"] = "?"
            annotated_item["valid"] = False
            annotated_item["expires_in"] = "unknown"
        annotated.append(annotated_item)
    return annotated


def get_cache_overview() -> tuple[list[dict[str, Any]], dict[str, list], str | None]:
    """Return cache tables, sample rows, and an optional error message."""
    try:
        if active_cache_backend() == "gcs":
            tables_info: list[dict[str, Any]] = []
            samples: dict[str, list] = {}
            for table in sorted(_ALLOWED_TABLES):
                entries = _gcs_iter_entries(table)
                total_rows = sum(len(entry.get("rows") or []) for entry in entries)
                counts: Counter[str] = Counter()
                meta = []
                sample_rows: list[dict[str, Any]] = []
                for entry in entries:
                    rows = list(entry.get("rows") or [])
                    meta.append(
                        {
                            "cache_key": entry.get("cache_key", ""),
                            "fetched_at": entry.get("fetched_at", ""),
                        }
                    )
                    for row in rows:
                        country = row.get("country")
                        if country:
                            counts[str(country)] += 1
                    if len(sample_rows) < 5:
                        sample_rows.extend(rows[: 5 - len(sample_rows)])

                if sample_rows:
                    samples[table] = sample_rows

                tables_info.append(
                    {
                        "name": table,
                        "total_rows": total_rows,
                        "ttl_hours": _TTL_HOURS.get(table, 24),
                        "country_rows": [
                            {"country": country, "rows": rows}
                            for country, rows in sorted(counts.items())
                        ],
                        "meta": _annotate_meta(table, meta),
                    }
                )

            return tables_info, samples, None

        con = _connect()
        tables_info = []
        samples = {}
        for table in sorted(_ALLOWED_TABLES):
            ttl_hours = _TTL_HOURS.get(table, 24)
            try:
                total = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except Exception:
                total = 0
            try:
                rows = con.execute(
                    f"SELECT country, COUNT(*) as rows FROM {table} GROUP BY country ORDER BY country"
                ).fetchdf().to_dict(orient="records")
            except Exception:
                rows = []
            try:
                meta = con.execute(
                    "SELECT cache_key, fetched_at FROM cache_meta WHERE table_name=? ORDER BY fetched_at DESC",
                    [table],
                ).fetchdf().to_dict(orient="records")
            except Exception:
                meta = []
            try:
                sample_rows = con.execute(f"SELECT * FROM {table} LIMIT 5").fetchdf().to_dict(orient="records")
                if sample_rows:
                    samples[table] = sample_rows
            except Exception:
                pass
            tables_info.append(
                {
                    "name": table,
                    "total_rows": total,
                    "ttl_hours": ttl_hours,
                    "country_rows": rows,
                    "meta": _annotate_meta(table, meta),
                }
            )
        return tables_info, samples, None
    except Exception as exc:
        return [], {}, str(exc)


def clear_cache() -> None:
    """Clear every cache entry from the active backend."""
    if active_cache_backend() == "gcs":
        bucket = _get_storage_client().bucket(_cache_bucket())
        for table in _ALLOWED_TABLES:
            prefix = f"{_cache_prefix()}/{table}/"
            blobs = list(bucket.list_blobs(prefix=prefix))
            if blobs:
                bucket.delete_blobs(blobs)
        return

    con = _connect()
    for table in _ALLOWED_TABLES:
        con.execute(f"DELETE FROM {table}")
    con.execute("DELETE FROM cache_meta")
