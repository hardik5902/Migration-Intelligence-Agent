"""Pydantic schemas for migration intelligence pipeline."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class DataPoint(BaseModel):
    model_config = ConfigDict(extra="ignore")

    claim: str
    value: str
    indicator: str = ""


class PushFactor(BaseModel):
    name: str
    pearson_r: float | None = None
    p_value: float | None = None
    description: str = ""


class ScoredDestination(BaseModel):
    country: str
    refugee_count: float | None = None
    share_of_outflow: float | None = None
    weighted_score: float | None = None


class NewsItem(BaseModel):
    title: str
    source: str = ""
    published_at: str = ""
    sentiment_score: float = 0.0


class ClimateSnapshot(BaseModel):
    avg_temp_anomaly_c: float = 0.0
    annual_precipitation_mm: float = 0.0
    extreme_heat_days_per_year: int = 0
    climate_risk_label: Literal["low", "moderate", "high", "severe"] = "moderate"


class SafetySummary(BaseModel):
    teleport_safety_score: float = 0.0
    acled_events_last_year: int = 0
    acled_fatalities_last_year: int = 0
    composite_safety_score: float = 0.0


class EmploymentData(BaseModel):
    unemployment_rate: float = 0.0
    youth_unemployment_rate: float = 0.0
    labor_force_participation: float = 0.0
    year: int = 0


class Citation(BaseModel):
    claim: str
    value: str
    source_api: str
    indicator_code: str = ""
    endpoint_url: str
    fetched_at: str


class CompetingHypothesis(BaseModel):
    hypothesis: str
    evidence_for: list[str] = Field(default_factory=list)
    evidence_against: list[str] = Field(default_factory=list)
    probability_score: float = 0.5


class ChartData(BaseModel):
    panel_title: str
    fig_json: str = ""
    caption: str = ""


class ChartPanel(BaseModel):
    fig_json: str
    caption: str
    data_sources: list[str] = Field(default_factory=list)


class ToolCall(BaseModel):
    tool_name: str
    params: dict[str, str] = Field(default_factory=dict)
    from_cache: bool = False
    started_at: str | None = None
    finished_at: str | None = None
    rows_returned: int | None = None
    endpoint_url: str | None = None
    source_api: str | None = None
    error: str | None = None


class IntentConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    intent: Literal[
        "push_factor",
        "destination",
        "historical",
        "real_time",
        "relocation_advisory",
    ] = "push_factor"
    country: str = ""
    country_code: str = ""
    target_country: str = ""
    target_country_code: str = ""
    year_from: int = 2010
    year_to: int = 2023
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "aqi": 0.4,
            "healthcare": 0.35,
            "education": 0.25,
            "safety": 0.3,
            "cost_of_living": 0.0,
        }
    )
    api_priority: list[str] = Field(default_factory=list)


class ToolSelection(BaseModel):
    """Smart router decision: which tools to call based on query analysis."""
    model_config = ConfigDict(extra="ignore")
    
    worldbank: bool = True
    unhcr: bool = True
    acled: bool = False
    teleport: bool = True
    news: bool = False
    climate: bool = True
    employment: bool = False
    aqi: bool = True
    reasoning: str = ""


class MigrationDataset(BaseModel):
    model_config = ConfigDict(extra="ignore")

    country: str = ""
    country_code: str = ""
    target_country: str = ""
    target_country_code: str = ""
    year_from: int = 2010
    year_to: int = 2023
    intent: str = "push_factor"
    displacement: list[dict[str, Any]] = Field(default_factory=list)
    destinations: list[dict[str, Any]] = Field(default_factory=list)
    worldbank: list[dict[str, Any]] = Field(default_factory=list)
    conflict_events: list[dict[str, Any]] = Field(default_factory=list)
    city_scores: list[dict[str, Any]] = Field(default_factory=list)
    news: list[dict[str, Any]] = Field(default_factory=list)
    gdelt: dict[str, Any] = Field(default_factory=dict)
    climate: list[dict[str, Any]] = Field(default_factory=list)
    employment: list[dict[str, Any]] = Field(default_factory=list)
    aqi: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    data_freshness: dict[str, str] = Field(default_factory=dict)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    top_headline: str | None = None
    missing_reasons: list[str] = Field(default_factory=list)


class PushFactorResult(BaseModel):
    top_driver: str = ""
    r: float | None = None
    granger_lag_years: int | None = None
    inflection_year: int | None = None
    p_value: float | None = None
    ranked_factors: list[PushFactor] = Field(default_factory=list)
    narrative: str = ""


class DestinationResult(BaseModel):
    top_destinations: list[ScoredDestination] = Field(default_factory=list)
    anomaly_note: str = ""
    narrative: str = ""


class PatternResult(BaseModel):
    template: str = "economic_collapse"
    similarity_to_syria: float | None = None
    similarity_to_zimbabwe: float | None = None
    news_lead_months: int | None = None
    narrative: str = ""


class RelocationRow(BaseModel):
    country: str
    weighted_score: float = 0.0
    pm25: float | None = None
    healthcare_signal: float | None = None
    education_signal: float | None = None
    safety_signal: float | None = None
    cost_of_living_signal: float | None = None
    salary_signal: float | None = None


class RelocationResult(BaseModel):
    active: bool = False
    top_countries: list[RelocationRow] = Field(default_factory=list)
    narrative: str = ""


class ToolSelectorOutput(BaseModel):
    """LLM output: which tools to call and which countries to compare."""
    model_config = ConfigDict(extra="ignore")

    selected_tools: list[str] = Field(default_factory=list)
    countries: list[str] = Field(default_factory=list)
    country_codes: list[str] = Field(default_factory=list)
    k: int = 5
    query_focus: str = ""
    year_from: int = 2015
    year_to: int = 2023
    reasoning: str = ""
    proxy_note: str = ""        # set when niche topic uses proxy indicators
    in_scope: bool = True       # False = query has no country/migration relevance
    out_of_scope_reason: str = ""  # human-readable explanation when in_scope=False
    worldbank_indicators: list[str] = Field(default_factory=list)  # specific WB indicators for this query


class Evidence(BaseModel):
    """A single LLM-generated evidence insight backed by collected data."""
    model_config = ConfigDict(extra="ignore")

    title: str = ""
    value: str = ""
    data_source: str = ""
    description: str = ""
    confidence: int = 50  # 0-100; assigned by LLM based on data completeness & recency


class HypothesisInsight(BaseModel):
    """A structured hypothesis grounded in EDA data with competing explanations."""
    model_config = ConfigDict(extra="ignore")

    headline: str = ""
    # 2-3 sentence summary with specific numbers
    summary: str = ""
    # The single anchoring data point (e.g. "GDP growth 1.8%, 2022")
    key_metric: str = ""
    # Explicit chain of reasoning from data → conclusion
    reasoning: str = ""
    # List of specific data points supporting the hypothesis (with values + years)
    evidence_for: list[str] = Field(default_factory=list)
    # Data points that complicate or challenge the hypothesis
    evidence_against: list[str] = Field(default_factory=list)
    # Alternative explanation the data could support
    competing_hypothesis: str = ""
    # Why the primary hypothesis is more supported than the alternative
    competing_verdict: str = ""
    data_source: str = ""
    confidence: int = 50


class CountryComparisonResult(BaseModel):
    """Final result of the country comparison pipeline."""
    model_config = ConfigDict(extra="ignore")

    query: str = ""
    countries: list[str] = Field(default_factory=list)
    country_codes: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    hypotheses: list[HypothesisInsight] = Field(default_factory=list)
    summary: str = ""
    chart_jsons: list[str] = Field(default_factory=list)
    query_focus: str = ""
    year_from: int = 2015
    year_to: int = 2023
    # Total data rows returned per tool across all countries (0 = tool returned nothing)
    tool_stats: dict[str, int] = Field(default_factory=dict)


class HypothesisReport(BaseModel):
    model_config = ConfigDict(extra="ignore")

    headline: str = ""
    supporting_points: list[DataPoint] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    competing_hypotheses: list[CompetingHypothesis] = Field(default_factory=list)
    primary_vs_alternatives: str = ""
    counter_argument: str = ""
    counter_rebuttal: str = ""
    confidence_score: float = 0.5
    top_push_factors: list[PushFactor] = Field(default_factory=list)
    top_destinations: list[ScoredDestination] = Field(default_factory=list)
    destination_insights: list[str] = Field(default_factory=list)
    historical_template: str = ""
    news_sentiment_summary: str = ""
    climate_risk_assessment: str = ""
    employment_outlook: str = ""
    safety_assessment: str = ""
    charts: list[ChartPanel] = Field(default_factory=list)
    recent_headlines: list[NewsItem] = Field(default_factory=list)
    data_freshness: dict[str, str] = Field(default_factory=dict)
