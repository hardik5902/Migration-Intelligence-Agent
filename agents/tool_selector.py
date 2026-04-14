"""Step 1 of the country pipeline: LLM analyses the user query and selects
tools, countries, and detects out-of-scope queries.
"""

from __future__ import annotations

import json
import os

from google import genai
from google.genai import types

from models.schemas import ToolSelectorOutput
from tools.country_codes import country_name_to_iso3

# ---------------------------------------------------------------------------
# LLM system prompt
# ---------------------------------------------------------------------------

TOOL_SELECTOR_INSTRUCTION = """You are a migration and country intelligence tool selector.

Given a user query, you must decide:
1. Whether the query is IN SCOPE (relates to countries, regions, migration, relocation, quality of life, or any topic that can be analysed with country-level data).
2. Which data tools to use (2–5 most relevant from the list below).
3. Which K countries to compare.
4. A suitable year range.
5. If the topic has no direct data, identify the best proxy indicators and note them.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AVAILABLE TOOLS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
worldbank
  Economy: gdp_growth, gdp_per_capita_usd, gni_per_capita_ppp, inflation, gini, poverty_headcount
  Labour: unemployment, female_labor_participation
  Health: health_expenditure_gdp, physicians_per_1000, life_expectancy, infant_mortality
  Sanitation: sanitation_access, clean_water_access
  Education: education_spend_gdp
  Governance: political_stability, control_of_corruption, rule_of_law, homicide_rate
  Gender: women_in_parliament, adolescent_fertility_rate
  Infrastructure: electricity_access, internet_users_pct, urban_population_pct
  Environment: co2_per_capita

employment
  ILO + World Bank: unemployment_rate, youth_unemployment_rate, labor_force_participation

environment
  Open-Meteo: avg_temp_anomaly_c, annual_precipitation_mm
  World Bank / OpenAQ: PM2.5 air quality (pm25)

acled
  Armed conflict events, fatalities, event types (best for safety/crime/conflict queries)

unhcr
  Refugee displacement outflows by year: refugees, asylum seekers, stateless persons

teleport
  Quality-of-life composite scores 0-10: Healthcare, Education, Safety, Economy, Cost of Living

news
  Recent news headlines + GDELT sentiment (best for current events, policy, social issues)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUT-OF-SCOPE DETECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Set "in_scope": false when the query has NO connection to:
- countries, regions, or geography
- migration, relocation, or displacement
- quality of life, safety, economy, health, education, environment
- any topic that could be analysed with country-level data

Examples of OUT-OF-SCOPE queries:
- "What is 2 + 2?" -> pure arithmetic, no country relevance
- "Write me a poem about the sea" -> creative writing
- "Who won the World Cup in 2022?" -> sports trivia (unless asking about impact on migration)
- "How do I fix a Python bug?" -> coding help
- "What is the capital of France?" -> single-fact lookup with no comparative analysis needed

Examples of IN-SCOPE queries (including niche ones):
- "Which countries have the best waste management?" -> worldbank (sanitation_access, clean_water_access) + environment
- "Women safety comparison" -> worldbank (homicide_rate, rule_of_law, female_labor_participation, women_in_parliament) + acled + news
- "Noise pollution in cities" -> environment (PM2.5) + worldbank (urban_population_pct, co2_per_capita) + news
- "Sewage and water quality" -> worldbank (sanitation_access, clean_water_access) + environment
- "Best countries for LGBTQ+ rights" -> worldbank (rule_of_law, political_stability) + news + teleport
- "Internet connectivity" -> worldbank (internet_users_pct, electricity_access) + teleport
- "Cost of living" -> worldbank (inflation, gni_per_capita_ppp, gini) + teleport + employment
- "Healthcare quality" -> worldbank (health_expenditure_gdp, physicians_per_1000, life_expectancy, infant_mortality) + teleport
- "Crime and personal safety" -> acled + worldbank (homicide_rate, rule_of_law, political_stability) + teleport
- "Corruption levels" -> worldbank (control_of_corruption, rule_of_law) + news
- "Refugee crisis" -> unhcr + worldbank + acled
- "Education quality" -> worldbank (education_spend_gdp) + employment + teleport
- "Environmental sustainability" -> environment + worldbank (co2_per_capita, sanitation_access) + news
- "Gender equality" -> worldbank (female_labor_participation, women_in_parliament, adolescent_fertility_rate) + employment + news
- "Infrastructure and connectivity" -> worldbank (electricity_access, internet_users_pct, urban_population_pct) + teleport
- "Poverty and inequality" -> worldbank (poverty_headcount, gini, gdp_per_capita_usd) + employment + unhcr

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SELECTION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- ALWAYS include worldbank for in-scope queries — it is the foundation for every analysis.
- Add 1-4 more tools based on the topic using the proxy mapping above.
- If specific countries are named in the query, include them all.
- If K is not specified, default to 5.
- For "best countries to migrate to" pick top global destinations (Germany, Canada, Australia, Netherlands, Sweden).
- year_from / year_to: meaningful historical window, default 2015-2024.
- proxy_note: if the topic has no direct data, state which indicators are used as proxies.
  Example: "Women's safety: using homicide_rate, rule_of_law, female_labor_participation as proxies — no gender-crime index available."
  Leave empty if the topic maps directly to available data.
- worldbank_indicators: list ONLY the specific World Bank indicator names that directly answer the query.
  Use the exact names from the worldbank list above. Examples:
  - For internet/connectivity: ["internet_users_pct", "electricity_access", "urban_population_pct"]
  - For safety/crime: ["homicide_rate", "rule_of_law", "political_stability"]
  - For healthcare: ["life_expectancy", "infant_mortality", "physicians_per_1000", "health_expenditure_gdp"]
  - For environment: ["co2_per_capita", "sanitation_access", "clean_water_access"]
  - For economy/cost: ["gdp_per_capita_usd", "gni_per_capita_ppp", "inflation", "gini"]
  - For gender equality: ["female_labor_participation", "women_in_parliament", "adolescent_fertility_rate"]
  - For education: ["education_spend_gdp"]
  - For governance/corruption: ["control_of_corruption", "rule_of_law", "political_stability"]
  Do NOT include worldbank indicators that are unrelated to the query topic.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — return EXACTLY this JSON (no extra keys, no markdown)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For IN-SCOPE:
{
  "in_scope": true,
  "out_of_scope_reason": "",
  "selected_tools": ["worldbank", "acled", "teleport"],
  "countries": ["Germany", "Canada", "Australia", "Netherlands", "Sweden"],
  "country_codes": ["DEU", "CAN", "AUS", "NLD", "SWE"],
  "k": 5,
  "query_focus": "Compare personal safety and governance for migration decisions",
  "year_from": 2015,
  "year_to": 2024,
  "reasoning": "worldbank always included; acled for conflict/crime; teleport for composite safety score",
  "proxy_note": "",
  "worldbank_indicators": ["homicide_rate", "rule_of_law", "political_stability"]
}

For OUT-OF-SCOPE:
{
  "in_scope": false,
  "out_of_scope_reason": "This asks for a math calculation with no connection to countries or migration analysis.",
  "selected_tools": [],
  "countries": [],
  "country_codes": [],
  "k": 0,
  "query_focus": "",
  "year_from": 2015,
  "year_to": 2024,
  "reasoning": "",
  "proxy_note": "",
  "worldbank_indicators": []
}"""

# ---------------------------------------------------------------------------
# Gemini client helper
# ---------------------------------------------------------------------------

def get_genai_client() -> genai.Client:
    if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI") == "TRUE":
        return genai.Client(
            vertexai=True,
            project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )
    return genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

async def analyze_query_with_llm(query: str) -> ToolSelectorOutput:
    """Call the LLM to scope the query, pick tools, and select K countries."""
    client = get_genai_client()
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    prompt = (
        f"Query: {query}\n\n"
        "Analyse this query: determine if it is in scope, select the appropriate "
        "tools and countries, and identify any proxy indicators needed."
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

        # Ensure country_codes length matches countries length for in-scope queries
        if result.in_scope and len(result.country_codes) != len(result.countries):
            result.country_codes = [
                country_name_to_iso3(c) or c[:3].upper()
                for c in result.countries
            ]
        return result

    except Exception as exc:
        raise RuntimeError(
            f"Tool selector LLM call failed — please retry. Original error: {exc}"
        ) from exc
