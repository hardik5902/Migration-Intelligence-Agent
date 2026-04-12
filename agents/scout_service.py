"""Parallel API collection + DuckDB cache → MigrationDataset."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import httpx

from models.schemas import Citation, IntentConfig, MigrationDataset
from tools import aqi_tools, climate_tools, employment_tools, news_tools, teleport_tools, unhcr_tools, worldbank_tools
from tools import acled_tools
from tools.country_codes import country_name_to_iso3, iso3_to_iso2
from tools.duckdb_tools import cache_key, replace_table_rows, run_sql_query, should_use_cache


async def collect_migration_dataset(intent: IntentConfig) -> MigrationDataset:
    """Fetch (or load TTL cache) all configured sources."""
    code = intent.country_code or country_name_to_iso3(intent.country) or "USA"
    name = intent.country or code
    y0, y1 = intent.year_from, intent.year_to
    ck = cache_key(code, y0, y1)

    citations: list[Citation] = []
    fresh: dict[str, str] = {}

    async with httpx.AsyncClient(timeout=120.0) as client:

        async def eco():
            if should_use_cache("economic_indicators", ck):
                rows = run_sql_query(
                    f"SELECT * FROM economic_indicators WHERE cache_key = '{ck}'"
                )
                return rows, []
            rows, urls = await worldbank_tools.fetch_macro_bundle(code, y0, y1, client)
            replace_table_rows(
                "economic_indicators",
                ck,
                rows,
                ["country", "year", "indicator", "value", "label", "endpoint_url", "fetched_at"],
            )
            if rows:
                fresh["World Bank"] = str(rows[0].get("fetched_at", ""))
            return rows, urls

        async def disp():
            if should_use_cache("displacement_data", ck):
                return run_sql_query(
                    f"SELECT * FROM displacement_data WHERE cache_key = '{ck}'"
                )
            rows, _url = await unhcr_tools.get_displacement_data(code, y0, y1, client)
            replace_table_rows(
                "displacement_data",
                ck,
                rows,
                [
                    "country",
                    "year",
                    "metric",
                    "value",
                    "coa",
                    "coa_name",
                    "endpoint_url",
                    "fetched_at",
                ],
            )
            if rows:
                fresh["UNHCR"] = str(rows[0].get("fetched_at", ""))
            return rows

        async def city():
            if should_use_cache("city_scores", ck):
                return run_sql_query(f"SELECT * FROM city_scores WHERE cache_key = '{ck}'")
            rows, _ = await teleport_tools.get_city_scores(name, client)
            replace_table_rows(
                "city_scores",
                ck,
                rows,
                ["slug", "category", "score_out_of_10", "endpoint_url", "fetched_at"],
            )
            if rows:
                fresh["Teleport"] = str(rows[0].get("fetched_at", ""))
            return rows

        async def news_c():
            if should_use_cache("news_articles", ck):
                return run_sql_query(f"SELECT * FROM news_articles WHERE cache_key = '{ck}'")
            rows, _ = await news_tools.get_country_news(name, None, client)
            replace_table_rows(
                "news_articles",
                ck,
                rows,
                ["country", "title", "source", "published_at", "endpoint_url", "fetched_at"],
            )
            if rows:
                fresh["NewsAPI"] = str(rows[0].get("fetched_at", ""))
            return rows

        async def clim():
            if should_use_cache("climate_data", ck):
                return run_sql_query(f"SELECT * FROM climate_data WHERE cache_key = '{ck}'")
            rows, _ = await climate_tools.get_climate_data(code, y0, y1, client)
            replace_table_rows(
                "climate_data",
                ck,
                rows,
                [
                    "country",
                    "year",
                    "avg_daily_max_temp_c",
                    "annual_precipitation_mm",
                    "endpoint_url",
                    "fetched_at",
                ],
            )
            if rows:
                fresh["Open-Meteo"] = str(rows[0].get("fetched_at", ""))
            return rows

        async def emp():
            if should_use_cache("employment_data", ck):
                return run_sql_query(f"SELECT * FROM employment_data WHERE cache_key = '{ck}'")
            rows, _ = await employment_tools.get_employment_data(code, y0, y1, client)
            replace_table_rows(
                "employment_data",
                ck,
                rows,
                ["country", "year", "unemployment_rate", "endpoint_url", "fetched_at"],
            )
            if rows:
                fresh["Employment"] = str(rows[0].get("fetched_at", ""))
            return rows

        async def conf():
            if should_use_cache("conflict_events", ck):
                return run_sql_query(f"SELECT * FROM conflict_events WHERE cache_key = '{ck}'")
            if intent.intent == "real_time":
                d1 = (datetime.now(timezone.utc) - timedelta(days=90)).strftime("%Y-%m-%d")
            else:
                d1 = f"{y0}-01-01"
            d2 = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            rows, _ = await acled_tools.get_conflict_events(code, d1, d2, client)
            replace_table_rows(
                "conflict_events",
                ck,
                rows,
                ["country", "event_date", "fatalities", "event_type", "endpoint_url", "fetched_at"],
            )
            if rows:
                fresh["ACLED"] = str(rows[0].get("fetched_at", ""))
            return rows

        async def aqi_local():
            if should_use_cache("aqi_data", ck):
                return run_sql_query(f"SELECT * FROM aqi_data WHERE cache_key = '{ck}'")
            iso2 = iso3_to_iso2(code) or code[:2]
            rows, _ = await aqi_tools.get_aqi_by_country(iso2, 40, client)
            for r in rows:
                r["country"] = r.get("country") or iso2
            replace_table_rows(
                "aqi_data",
                ck,
                rows,
                ["country", "location", "pm25", "endpoint_url", "fetched_at"],
            )
            if rows:
                fresh["OpenAQ"] = str(rows[0].get("fetched_at", ""))
            return rows

        core = [eco(), disp(), city(), news_c(), clim(), emp()]
        extra = []
        if intent.intent in ("push_factor", "real_time", "historical"):
            extra.append(conf())
        if intent.intent in ("relocation_advisory", "real_time"):
            extra.append(aqi_local())

        bundled = await asyncio.gather(*core, *extra, return_exceptions=True)
        n_core = len(core)
        core_res = bundled[:n_core]
        extra_res = bundled[n_core:]

        world_rows, _wb_urls = _unwrap_pair(core_res[0])
        disp_rows = _unwrap_list(core_res[1])
        city_rows = _unwrap_list(core_res[2])
        news_rows = _unwrap_list(core_res[3])
        climate_rows = _unwrap_list(core_res[4])
        emp_rows = _unwrap_list(core_res[5])

        conflict_rows: list = []
        aqi_rows: list = []
        ei = 0
        if intent.intent in ("push_factor", "real_time", "historical"):
            conflict_rows = _unwrap_list(extra_res[ei])
            ei += 1
        if intent.intent in ("relocation_advisory", "real_time"):
            aqi_rows = _unwrap_list(extra_res[ei])

    gd, _gurl = await news_tools.get_gdelt_sentiment(name, None)

    extra_disp: list = []
    if intent.intent == "historical":
        for extra in ("SYR", "ZWE"):
            if extra == code:
                continue
            ck2 = cache_key(extra, y0, y1)
            if not should_use_cache("displacement_data", ck2):
                async with httpx.AsyncClient(timeout=90.0) as c2:
                    r2, _ = await unhcr_tools.get_displacement_data(extra, y0, y1, c2)
                    replace_table_rows(
                        "displacement_data",
                        ck2,
                        r2,
                        [
                            "country",
                            "year",
                            "metric",
                            "value",
                            "coa",
                            "coa_name",
                            "endpoint_url",
                            "fetched_at",
                        ],
                    )
            extra_disp.extend(
                run_sql_query(f"SELECT * FROM displacement_data WHERE cache_key = '{ck2}'")
            )

    if world_rows:
        citations.append(
            Citation(
                claim=f"Macro indicators for {code}",
                value=str(len(world_rows)),
                source_api="World Bank API",
                indicator_code="bundle",
                endpoint_url=str(world_rows[0].get("endpoint_url", "")),
                fetched_at=str(world_rows[0].get("fetched_at", "")),
            )
        )
    if disp_rows:
        citations.append(
            Citation(
                claim="UNHCR displacement rows",
                value=str(len(disp_rows)),
                source_api="UNHCR API",
                indicator_code="population/v1",
                endpoint_url=str(disp_rows[0].get("endpoint_url", "")),
                fetched_at=str(disp_rows[0].get("fetched_at", "")),
            )
        )

    destinations: list[dict] = []
    if disp_rows:
        df_co: dict[str, float] = {}
        for r in disp_rows:
            dest = str(r.get("coa_name") or r.get("coa") or "unknown")
            df_co[dest] = df_co.get(dest, 0.0) + float(r.get("value") or 0)
        for k, v in sorted(df_co.items(), key=lambda x: -x[1])[:15]:
            destinations.append({"destination_country": k, "refugee_count": v})

    if intent.intent == "relocation_advisory":
        refs = ["NZ", "FI", "NO", "SE", "CA", "DE"]
        iso2o = iso3_to_iso2(code) or "IN"
        more: list = []
        async with httpx.AsyncClient(timeout=90.0) as c3:
            for iso in [iso2o] + refs:
                ckx = cache_key(iso, y0, y1)
                if should_use_cache("aqi_data", ckx):
                    more.extend(run_sql_query(f"SELECT * FROM aqi_data WHERE cache_key = '{ckx}'"))
                else:
                    rows, _ = await aqi_tools.get_aqi_by_country(iso, 25, c3)
                    for r in rows:
                        r["country"] = r.get("country") or iso
                    replace_table_rows(
                        "aqi_data",
                        ckx,
                        rows,
                        ["country", "location", "pm25", "endpoint_url", "fetched_at"],
                    )
                    more.extend(rows)
        aqi_rows = more

    ds = MigrationDataset(
        country=name,
        country_code=code,
        year_from=y0,
        year_to=y1,
        intent=intent.intent,
        displacement=disp_rows + extra_disp,
        destinations=destinations,
        worldbank=world_rows,
        conflict_events=conflict_rows,
        city_scores=city_rows,
        news=news_rows,
        gdelt=gd,
        climate=climate_rows,
        employment=emp_rows,
        aqi=aqi_rows,
        citations=citations,
        data_freshness={**fresh, "GDELT": str(gd.get("fetched_at", ""))},
    )
    return ds


def _unwrap_pair(res):
    if isinstance(res, Exception):
        return [], []
    return res[0], res[1]


def _unwrap_list(res):
    if isinstance(res, Exception):
        return []
    return res
