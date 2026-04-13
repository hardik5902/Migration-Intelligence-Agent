"""Parallel API collection + DuckDB cache → MigrationDataset."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import httpx

from models.schemas import Citation, IntentConfig, MigrationDataset, ToolCall, ToolSelection
from tools import aqi_tools, climate_tools, employment_tools, news_tools, teleport_tools, unhcr_tools, worldbank_tools
from tools import acled_tools
from tools.country_codes import country_name_to_iso3, iso3_to_iso2, iso3_to_name
from tools.duckdb_tools import cache_key, replace_table_rows, run_sql_query, should_use_cache
from agents.progress_tracker import get_tracker


async def collect_migration_dataset(
    intent: IntentConfig,
    tool_selection: dict | None = None,
) -> MigrationDataset:
    """Fetch (or load TTL cache) selected sources based on tool_selection."""
    # Parse tool selection (default: all on)
    tools = {
        "worldbank": True,
        "unhcr": True,
        "acled": True,
        "teleport": True,
        "news": True,
        "climate": True,
        "employment": True,
        "aqi": True,
    }
    if tool_selection:
        tools.update(tool_selection)
    code = intent.country_code or country_name_to_iso3(intent.country) or ""
    name = intent.country or iso3_to_name(code) or code
    target_code = intent.target_country_code or country_name_to_iso3(intent.target_country) or ""
    target_name = intent.target_country or iso3_to_name(target_code) or target_code
    y0, y1 = intent.year_from, intent.year_to
    ck = cache_key(code, y0, y1)

    if not code and intent.intent != "relocation_advisory":
        return MigrationDataset(
            country=name,
            country_code=code,
            target_country=target_name,
            target_country_code=target_code,
            year_from=y0,
            year_to=y1,
            intent=intent.intent,
        )

    citations: list[Citation] = []
    fresh: dict[str, str] = {}
    tool_calls: list[ToolCall] = []

    async with httpx.AsyncClient(timeout=120.0) as client:

        async def eco():
            started = datetime.now(timezone.utc).isoformat()
            try:
                if should_use_cache("economic_indicators", ck):
                    rows = run_sql_query(
                        f"SELECT * FROM economic_indicators WHERE cache_key = '{ck}'"
                    )
                    from_cache = True
                    urls = []
                else:
                    rows, urls = await worldbank_tools.fetch_macro_bundle(code, y0, y1, client)
                    # fetch_macro_bundle may append ERROR:... entries into urls for
                    # per-indicator diagnostics; separate them out into an error string
                    errors = [u for u in urls if isinstance(u, str) and u.startswith("ERROR:")]
                    # keep only real endpoint urls
                    urls = [u for u in urls if not (isinstance(u, str) and u.startswith("ERROR:"))]
                    err_str = "; ".join(errors) if errors else None
                    from_cache = False
                    replace_table_rows(
                        "economic_indicators",
                        ck,
                        rows,
                        ["country", "year", "indicator", "value", "label", "endpoint_url", "fetched_at"],
                    )
                if rows:
                    fresh["World Bank"] = str(rows[0].get("fetched_at", ""))
                return rows, urls
            finally:
                finished = datetime.now(timezone.utc).isoformat()
                tool_calls.append(
                    ToolCall(
                        tool_name="worldbank.fetch_macro_bundle",
                        params={"country_code": code, "year_from": str(y0), "year_to": str(y1)},
                        from_cache=locals().get('from_cache', False),
                        started_at=started,
                        finished_at=finished,
                        rows_returned=(len(rows) if 'rows' in locals() and rows else 0),
                        endpoint_url=(rows[0].get('endpoint_url') if 'rows' in locals() and rows else None),
                        source_api="World Bank API",
                        error=locals().get('err_str', None) or None,
                    )
                )

        async def disp():
            started = datetime.now(timezone.utc).isoformat()
            try:
                if should_use_cache("displacement_data", ck):
                    rows = run_sql_query(f"SELECT * FROM displacement_data WHERE cache_key = '{ck}'")
                    from_cache = True
                else:
                    rows, _url = await unhcr_tools.get_displacement_data(code, y0, y1, client)
                    from_cache = False
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
            finally:
                finished = datetime.now(timezone.utc).isoformat()
                tool_calls.append(
                    ToolCall(
                        tool_name="unhcr.get_displacement_data",
                        params={"country_code": code, "year_from": str(y0), "year_to": str(y1)},
                        from_cache=locals().get('from_cache', False),
                        started_at=started,
                        finished_at=finished,
                        rows_returned=(len(rows) if 'rows' in locals() and rows else 0),
                        endpoint_url=(rows[0].get('endpoint_url') if 'rows' in locals() and rows else None),
                        source_api="UNHCR API",
                    )
                )

        async def city():
            started = datetime.now(timezone.utc).isoformat()
            try:
                if should_use_cache("city_scores", ck):
                    rows = run_sql_query(f"SELECT * FROM city_scores WHERE cache_key = '{ck}'")
                    from_cache = True
                else:
                    rows, _ = await teleport_tools.get_city_scores(name, country_iso3=code, client=client)
                    from_cache = False
                    replace_table_rows(
                        "city_scores",
                        ck,
                        rows,
                        ["country", "slug", "category", "score_out_of_10", "endpoint_url", "fetched_at"],
                    )
                if rows:
                    fresh["Teleport"] = str(rows[0].get("fetched_at", ""))
                return rows
            finally:
                finished = datetime.now(timezone.utc).isoformat()
                tool_calls.append(
                    ToolCall(
                        tool_name="teleport.get_city_scores",
                        params={"query": name},
                        from_cache=locals().get('from_cache', False),
                        started_at=started,
                        finished_at=finished,
                        rows_returned=(len(rows) if 'rows' in locals() and rows else 0),
                        endpoint_url=(rows[0].get('endpoint_url') if 'rows' in locals() and rows else None),
                        source_api="Teleport API",
                    )
                )

        async def news_c():
            started = datetime.now(timezone.utc).isoformat()
            try:
                if should_use_cache("news_articles", ck):
                    rows = run_sql_query(f"SELECT * FROM news_articles WHERE cache_key = '{ck}'")
                    from_cache = True
                else:
                    rows, _ = await news_tools.get_country_news(name, None, client)
                    from_cache = False
                    replace_table_rows(
                        "news_articles",
                        ck,
                        rows,
                        ["country", "title", "source", "published_at", "endpoint_url", "fetched_at"],
                    )
                if rows:
                    fresh["NewsAPI"] = str(rows[0].get("fetched_at", ""))
                return rows
            finally:
                finished = datetime.now(timezone.utc).isoformat()
                tool_calls.append(
                    ToolCall(
                        tool_name="news.get_country_news",
                        params={"query": name},
                        from_cache=locals().get('from_cache', False),
                        started_at=started,
                        finished_at=finished,
                        rows_returned=(len(rows) if 'rows' in locals() and rows else 0),
                        endpoint_url=(rows[0].get('endpoint_url') if 'rows' in locals() and rows else None),
                        source_api="NewsAPI",
                    )
                )

        async def clim():
            started = datetime.now(timezone.utc).isoformat()
            try:
                if should_use_cache("climate_data", ck):
                    rows = run_sql_query(f"SELECT * FROM climate_data WHERE cache_key = '{ck}'")
                    from_cache = True
                else:
                    rows, _ = await climate_tools.get_climate_data(code, y0, y1, client)
                    from_cache = False
                    replace_table_rows(
                        "climate_data",
                        ck,
                        rows,
                        [
                            "country",
                            "year",
                            "avg_daily_max_temp_c",
                            "annual_precipitation_mm",
                            "avg_temp_anomaly_c",
                            "extreme_heat_days",
                            "endpoint_url",
                            "fetched_at",
                        ],
                    )
                if rows:
                    fresh["Open-Meteo"] = str(rows[0].get("fetched_at", ""))
                return rows
            finally:
                finished = datetime.now(timezone.utc).isoformat()
                tool_calls.append(
                    ToolCall(
                        tool_name="open-meteo.get_climate_data",
                        params={"country_code": code, "year_from": str(y0), "year_to": str(y1)},
                        from_cache=locals().get('from_cache', False),
                        started_at=started,
                        finished_at=finished,
                        rows_returned=(len(rows) if 'rows' in locals() and rows else 0),
                        endpoint_url=(rows[0].get('endpoint_url') if 'rows' in locals() and rows else None),
                        source_api="Open-Meteo",
                    )
                )

        async def emp():
            started = datetime.now(timezone.utc).isoformat()
            try:
                if should_use_cache("employment_data", ck):
                    rows = run_sql_query(f"SELECT * FROM employment_data WHERE cache_key = '{ck}'")
                    from_cache = True
                else:
                    rows, _ = await employment_tools.get_employment_data(code, y0, y1, client)
                    from_cache = False
                    replace_table_rows(
                        "employment_data",
                        ck,
                        rows,
                        [
                            "country",
                            "year",
                            "unemployment_rate",
                            "youth_unemployment_rate",
                            "labor_force_participation",
                            "endpoint_url",
                            "fetched_at",
                        ],
                    )
                if rows:
                    fresh["Employment"] = str(rows[0].get("fetched_at", ""))
                return rows
            finally:
                finished = datetime.now(timezone.utc).isoformat()
                tool_calls.append(
                    ToolCall(
                        tool_name="ilo.get_employment_data",
                        params={"country_code": code, "year_from": str(y0), "year_to": str(y1)},
                        from_cache=locals().get('from_cache', False),
                        started_at=started,
                        finished_at=finished,
                        rows_returned=(len(rows) if 'rows' in locals() and rows else 0),
                        endpoint_url=(rows[0].get('endpoint_url') if 'rows' in locals() and rows else None),
                        source_api="ILOSTAT",
                    )
                )

        async def conf():
            if should_use_cache("conflict_events", ck):
                return run_sql_query(f"SELECT * FROM conflict_events WHERE cache_key = '{ck}'")
            if intent.intent == "real_time":
                d1 = (datetime.now(timezone.utc) - timedelta(days=90)).strftime("%Y-%m-%d")
            else:
                d1 = f"{y0}-01-01"
            d2 = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            started = datetime.now(timezone.utc).isoformat()
            try:
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
            finally:
                finished = datetime.now(timezone.utc).isoformat()
                tool_calls.append(
                    ToolCall(
                        tool_name="acled.get_conflict_events",
                        params={"country_code": code, "date_from": d1, "date_to": d2},
                        from_cache=False,
                        started_at=started,
                        finished_at=finished,
                        rows_returned=(len(rows) if 'rows' in locals() and rows else 0),
                        endpoint_url=(rows[0].get('endpoint_url') if 'rows' in locals() and rows else None),
                        source_api="ACLED API",
                    )
                )

        async def aqi_local():
            started = datetime.now(timezone.utc).isoformat()
            try:
                if should_use_cache("aqi_data", ck):
                    rows = run_sql_query(f"SELECT * FROM aqi_data WHERE cache_key = '{ck}'")
                    from_cache = True
                else:
                    iso2 = iso3_to_iso2(code) or code[:2]
                    rows, _ = await aqi_tools.get_aqi_by_country(code, country_iso2=iso2, top_n=40, client=client)
                    for r in rows:
                        r["country"] = r.get("country") or iso2
                    from_cache = False
                    replace_table_rows(
                        "aqi_data",
                        ck,
                        rows,
                        ["country", "location", "city", "pm25", "endpoint_url", "fetched_at"],
                    )
                if rows:
                    fresh["OpenAQ"] = str(rows[0].get("fetched_at", ""))
                return rows
            finally:
                finished = datetime.now(timezone.utc).isoformat()
                tool_calls.append(
                    ToolCall(
                        tool_name="openaq.get_aqi_by_country",
                        params={"country_iso2": iso3_to_iso2(code) or code[:2]},
                        from_cache=locals().get('from_cache', False),
                        started_at=started,
                        finished_at=finished,
                        rows_returned=(len(rows) if 'rows' in locals() and rows else 0),
                        endpoint_url=(rows[0].get('endpoint_url') if 'rows' in locals() and rows else None),
                        source_api="OpenAQ",
                    )
                )

        # Build task dict: only execute enabled tools
        tracker = get_tracker()
        tasks_dict = {}
        
        if tools["worldbank"]:
            tracker.log_tool_call("World Bank", {"country_code": code, "years": f"{y0}-{y1}"})
            tasks_dict["eco"] = eco()
        if tools["unhcr"]:
            tracker.log_tool_call("UNHCR", {"country_code": code, "years": f"{y0}-{y1}"})
            tasks_dict["disp"] = disp()
        if tools["teleport"]:
            tracker.log_tool_call("Teleport", {"query": name})
            tasks_dict["city"] = city()
        if tools["news"]:
            tracker.log_tool_call("NewsAPI", {"query": name})
            tasks_dict["news_c"] = news_c()
        if tools["climate"]:
            tracker.log_tool_call("Climate", {"country_code": code, "years": f"{y0}-{y1}"})
            tasks_dict["clim"] = clim()
        if tools["employment"]:
            tracker.log_tool_call("Employment", {"country_code": code, "years": f"{y0}-{y1}"})
            tasks_dict["emp"] = emp()
        if tools["acled"]:
            tracker.log_tool_call("ACLED", {"country_code": code, "date_range": f"{y0}-{y1}"})
            tasks_dict["conf"] = conf()
        if tools["aqi"]:
            tracker.log_tool_call("AQI", {"country_iso2": iso3_to_iso2(code) or code[:2]})
            tasks_dict["aqi_local"] = aqi_local()

        # Execute enabled tasks
        task_names = list(tasks_dict.keys())
        task_coros = list(tasks_dict.values())
        tracker.log_step(
            stage="data_collection",
            status="started",
            details=f"Collecting data from {len(task_names)} sources: {', '.join(task_names)}",
            metadata={"tools": task_names, "country": name, "code": code},
        )
        bundled = await asyncio.gather(*task_coros, return_exceptions=True) if task_coros else []
        tracker.log_step(
            stage="data_collection",
            status="completed",
            details=f"Collected data from {len(bundled)} sources",
            metadata={"source_count": len(bundled)},
        )

        # Unpack results using task names
        results = {name: bundled[i] for i, name in enumerate(task_names)}

        world_rows, _wb_urls = _unwrap_pair(results.get("eco")) if "eco" in results else ([], [])
        disp_rows = _unwrap_list(results.get("disp")) if "disp" in results else []
        city_rows = _unwrap_list(results.get("city")) if "city" in results else []
        news_rows = _unwrap_list(results.get("news_c")) if "news_c" in results else []
        climate_rows = _unwrap_list(results.get("clim")) if "clim" in results else []
        emp_rows = _unwrap_list(results.get("emp")) if "emp" in results else []
        conflict_rows = _unwrap_list(results.get("conf")) if "conf" in results else []
        aqi_rows = _unwrap_list(results.get("aqi_local")) if "aqi_local" in results else []

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

    # Fetch top headline for the top suggested destination (if any)
    top_headline: str | None = None
    if destinations:
        suggested = destinations[0]["destination_country"]
        try:
            top_news_rows, _ = await news_tools.get_country_news(suggested, None, None)
            if top_news_rows:
                top_headline = top_news_rows[0].get("title") or None
            tool_calls.append(
                ToolCall(
                    tool_name="news.get_country_news",
                    params={"query": suggested},
                    from_cache=False,
                    started_at=datetime.now(timezone.utc).isoformat(),
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    rows_returned=(len(top_news_rows) if top_news_rows else 0),
                    endpoint_url=(top_news_rows[0].get("endpoint_url") if top_news_rows else None),
                    source_api="NewsAPI",
                )
            )
        except Exception:
            # don't fail the whole collection for headline fetch
            pass

    if intent.intent == "relocation_advisory":
        refs = ["NZL", "FIN", "NOR", "SWE", "CAN", "DEU", "NLD", "CHE", "DNK", "IRL", "AUS", "SGP"]
        more: list = []
        extra_world: list = []
        extra_city: list = []
        async with httpx.AsyncClient(timeout=90.0) as c3:
            for candidate in [code] + refs:
                iso2 = iso3_to_iso2(candidate) or candidate[:2]
                candidate_name = iso3_to_name(candidate) or candidate
                ckx = cache_key(candidate, y0, y1)
                if should_use_cache("aqi_data", ckx):
                    more.extend(run_sql_query(f"SELECT * FROM aqi_data WHERE cache_key = '{ckx}'"))
                else:
                    rows, _ = await aqi_tools.get_aqi_by_country(candidate, country_iso2=iso2, top_n=25, client=c3)
                    for r in rows:
                        r["country"] = r.get("country") or candidate
                    replace_table_rows(
                        "aqi_data",
                        ckx,
                        rows,
                        ["country", "location", "city", "pm25", "endpoint_url", "fetched_at"],
                    )
                    more.extend(rows)
                if should_use_cache("city_scores", ckx):
                    extra_city.extend(run_sql_query(f"SELECT * FROM city_scores WHERE cache_key = '{ckx}'"))
                else:
                    city_rows, _ = await teleport_tools.get_city_scores(candidate_name, country_iso3=candidate, client=c3)
                    replace_table_rows(
                        "city_scores",
                        ckx,
                        city_rows,
                        ["country", "slug", "category", "score_out_of_10", "endpoint_url", "fetched_at"],
                    )
                    extra_city.extend(city_rows)
                if should_use_cache("economic_indicators", ckx):
                    extra_world.extend(run_sql_query(f"SELECT * FROM economic_indicators WHERE cache_key = '{ckx}'"))
                else:
                    wb_rows, _ = await worldbank_tools.fetch_relocation_bundle(candidate, y0, y1, c3)
                    replace_table_rows(
                        "economic_indicators",
                        ckx,
                        wb_rows,
                        ["country", "year", "indicator", "value", "label", "endpoint_url", "fetched_at"],
                    )
                    extra_world.extend(wb_rows)
        aqi_rows = more
        city_rows.extend(extra_city)
        world_rows.extend(extra_world)

    # Compute missing reasons for diagnostic reporting
    missing_reasons: list[str] = []
    if not world_rows:
        missing_reasons.append("no_economic_indicators")
    if not disp_rows:
        missing_reasons.append("no_displacement_data")
    if not news_rows:
        missing_reasons.append("no_news_headlines")
    if not climate_rows:
        missing_reasons.append("no_climate_data")
    if not emp_rows:
        missing_reasons.append("no_employment_data")
    if intent.intent in ("relocation_advisory", "real_time") and not aqi_rows:
        missing_reasons.append("no_aqi_data")

    ds = MigrationDataset(
        country=name,
        country_code=code,
        target_country=target_name,
        target_country_code=target_code,
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
        tool_calls=tool_calls,
        top_headline=top_headline,
        missing_reasons=missing_reasons,
    )
    
    # Debug output
    print(f"\n[SCOUT] Data Collection Complete for {name} ({code}):")
    print(f"  Displacement: {len(disp_rows + extra_disp)} rows")
    print(f"  World Bank: {len(world_rows)} rows")
    print(f"  Conflict Events: {len(conflict_rows)} rows")
    print(f"  Climate: {len(climate_rows)} rows")
    print(f"  AQI: {len(aqi_rows)} rows")
    print(f"  News: {len(news_rows)} rows")
    print(f"  Employment: {len(emp_rows)} rows")
    print(f"  City Scores: {len(city_rows)} rows")
    print(f"  Destinations: {len(destinations)} (from {len(disp_rows)} UNHCR rows)")
    print(f"  Tool calls logged: {len(tool_calls)}")
    print(f"  Missing data: {missing_reasons or 'none'}")
    print()
    
    return ds


def _unwrap_pair(res):
    if isinstance(res, Exception):
        return [], []
    return res[0], res[1]


def _unwrap_list(res):
    if isinstance(res, Exception):
        return []
    return res
