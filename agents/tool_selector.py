"""Step 1 of the country pipeline: LLM analyses the query and proposes tools.

Country selection is hybrid:
- explicit country mentions are extracted deterministically
- the LLM proposes candidate countries when the query is broad
- final countries are later validated against returned data coverage
"""

from __future__ import annotations

import json
import os

from google import genai
from google.genai import types

from agents.pipeline_config import HIGH_COVERAGE_DEFAULTS, TOOL_DEFAULT_COUNTRY_POOLS
from models.schemas import ToolSelectorOutput
from tools.country_codes import (
    all_country_names,
    country_name_to_iso3,
    iso3_to_name,
    normalize_name,
)

TOOL_SELECTOR_INSTRUCTION = """You are a migration and country intelligence tool selector.

Given a user query, return JSON describing:
1. whether the query is in scope
2. which tools are needed
3. which countries are reasonable candidate comparisons
4. a meaningful year range
5. the exact World Bank indicators relevant to the topic
6. the downstream country strategy

Rules:
- Always include worldbank for in-scope queries.
- Select 2-5 tools total.
- `countries` is only a candidate list. The application will deterministically
  extract explicit countries from the query and later validate final countries
  against actual data coverage.
- Set `country_strategy` to one of:
  - "explicit_only": the query directly names comparison countries
  - "mixed": some countries are named, but broad candidate suggestions are useful
  - "high_coverage_defaults": the query is broad and should use strong
    public-data-coverage defaults
- If the query is broad, prefer high-coverage countries such as Germany, Canada,
  Australia, Netherlands, and Sweden unless the topic suggests a different pool.
- `worldbank_indicators` must contain only indicators directly relevant to the query.
- Return only JSON, with no markdown.

Available tools:
- worldbank
- employment
- environment
- acled
- unhcr
- teleport
- news

For in-scope queries, return:
{
  "in_scope": true,
  "out_of_scope_reason": "",
  "selected_tools": ["worldbank", "teleport", "employment"],
  "countries": ["Germany", "Canada", "Australia", "Netherlands", "Sweden"],
  "country_codes": ["DEU", "CAN", "AUS", "NLD", "SWE"],
  "country_strategy": "high_coverage_defaults",
  "k": 5,
  "query_focus": "Compare healthcare and education for migration decisions",
  "year_from": 2015,
  "year_to": 2024,
  "reasoning": "worldbank for baseline indicators, teleport for quality-of-life composites, employment for labour market context",
  "proxy_note": "",
  "worldbank_indicators": ["life_expectancy", "infant_mortality", "physicians_per_1000", "education_spend_gdp"]
}

For out-of-scope queries, return:
{
  "in_scope": false,
  "out_of_scope_reason": "This query does not relate to countries, migration, or country-level quality-of-life analysis.",
  "selected_tools": [],
  "countries": [],
  "country_codes": [],
  "country_strategy": "explicit_only",
  "k": 0,
  "query_focus": "",
  "year_from": 2015,
  "year_to": 2024,
  "reasoning": "",
  "proxy_note": "",
  "worldbank_indicators": []
}"""


def get_genai_client() -> genai.Client:
    if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI") == "TRUE":
        return genai.Client(
            vertexai=True,
            project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )
    return genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))


def _dedupe_country_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for name, code in pairs:
        normalized_code = (code or "").strip().upper()
        if len(normalized_code) != 3 or normalized_code in seen:
            continue
        seen.add(normalized_code)
        out.append((name, normalized_code))
    return out


def _extract_explicit_countries(query: str) -> list[tuple[str, str]]:
    normalized_query = f" {normalize_name(query)} "
    pairs: list[tuple[str, str]] = []
    for country_name in all_country_names():
        code = country_name_to_iso3(country_name)
        if not code:
            continue
        if f" {country_name} " in normalized_query:
            pairs.append((iso3_to_name(code) or country_name.title(), code))
    return _dedupe_country_pairs(pairs)


def _normalize_country_suggestions(
    suggested_names: list[str],
    suggested_codes: list[str],
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    for code in suggested_codes:
        normalized_code = (code or "").strip().upper()
        if len(normalized_code) == 3:
            pairs.append((iso3_to_name(normalized_code) or normalized_code, normalized_code))

    for name in suggested_names:
        code = country_name_to_iso3(name or "")
        if code:
            pairs.append((iso3_to_name(code) or name.strip().title(), code))

    return _dedupe_country_pairs(pairs)


def _default_country_codes_for_tools(selected_tools: list[str], k: int) -> list[str]:
    ranked_codes: list[str] = []
    for tool in selected_tools:
        for code in TOOL_DEFAULT_COUNTRY_POOLS.get(tool, []):
            if code not in ranked_codes:
                ranked_codes.append(code)
    for code in HIGH_COVERAGE_DEFAULTS:
        if code not in ranked_codes:
            ranked_codes.append(code)
    return ranked_codes[: max(k, 3)]


def _resolve_candidate_countries(
    query: str,
    selector: ToolSelectorOutput,
) -> ToolSelectorOutput:
    explicit_pairs = _extract_explicit_countries(query)
    suggested_pairs = _normalize_country_suggestions(
        selector.countries,
        selector.country_codes,
    )

    if explicit_pairs:
        resolved_pairs = explicit_pairs[: selector.k]
        if selector.country_strategy == "mixed":
            current_codes = {code for _, code in resolved_pairs}
            for name, code in suggested_pairs:
                if len(resolved_pairs) >= selector.k:
                    break
                if code not in current_codes:
                    current_codes.add(code)
                    resolved_pairs.append((name, code))
        selector.country_strategy = (
            "explicit_only" if len(resolved_pairs) == len(explicit_pairs[: selector.k]) else "mixed"
        )
    elif suggested_pairs:
        resolved_pairs = suggested_pairs[: selector.k]
        if not selector.country_strategy:
            selector.country_strategy = "mixed"
    else:
        resolved_pairs = [
            (iso3_to_name(code) or code, code)
            for code in _default_country_codes_for_tools(selector.selected_tools, selector.k)
        ]
        selector.country_strategy = "high_coverage_defaults"

    selector.countries = [name for name, _ in resolved_pairs]
    selector.country_codes = [code for _, code in resolved_pairs]
    selector.k = len(selector.countries)
    return selector


async def analyze_query_with_llm(query: str) -> ToolSelectorOutput:
    """Call the LLM to scope the query, pick tools, and propose candidates."""
    client = get_genai_client()
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    prompt = (
        f"Query: {query}\n\n"
        "Analyse this query: determine if it is in scope, select the appropriate "
        "tools, propose candidate countries, and identify any proxy indicators needed."
    )

    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=TOOL_SELECTOR_INSTRUCTION,
                response_mime_type="application/json",
                temperature=0.1,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        raw = json.loads(response.text)
        result = ToolSelectorOutput.model_validate(raw)

        if result.in_scope:
            result = _resolve_candidate_countries(query, result)
        else:
            result.countries = []
            result.country_codes = []

        return result

    except Exception as exc:
        raise RuntimeError(
            f"Tool selector LLM call failed - please retry. Original error: {exc}"
        ) from exc
