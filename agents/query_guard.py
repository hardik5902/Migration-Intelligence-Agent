"""Prompt-injection guard for user queries.

Detects and rejects queries that attempt to:
  - Override system instructions ("ignore previous instructions", "forget …")
  - Request synthetic / fabricated data ("generate synthetic", "make up data")
  - Suppress citations, schemas, or data sources
  - Leak the system prompt
  - Force a plain-text / schema-bypass response
  - Restrict analysis to specific topics while ignoring others

Returns a (safe: bool, reason: str) tuple.  When safe=False, the reason is
displayed to the user and the pipeline is not started.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Injection pattern catalogue
# Each entry: (compiled regex, human-readable category label)
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Instruction override
    (re.compile(
        r"\b(ignore|forget|disregard|override|bypass|skip)\b.{0,40}"
        r"\b(previous|prior|above|system|all|instructions?|rules?|constraints?|prompt)\b",
        re.I,
    ), "instruction override"),

    # System-prompt / internals leak
    (re.compile(
        r"\b(reveal|show|print|output|display|repeat|expose|dump)\b.{0,40}"
        r"\b(system\s*prompt|system\s*config\w*|instructions?|internals?|rules?)\b",
        re.I,
    ), "system-prompt extraction"),

    # Synthetic / fabricated data
    (re.compile(
        r"\b(generat|creat|fabricat|invent|simulat|syntheti[zs]|make\s*up|produc)\w*\b.{0,50}"
        r"\b(data|numbers?|statistics?|results?|figures?|values?|analysis)\b",
        re.I,
    ), "synthetic data request"),

    # "Make it look real / convincing" framing
    (re.compile(
        r"\b(convincing|realistic|look\s+like\s+real|as\s+if\s+(it\s+came|from|real))\b",
        re.I,
    ), "fabrication framing"),

    # Hide / suppress citations or data sources
    (re.compile(
        r"\b(hide|suppress|omit|remove|don[''`]?t\s+(show|include|use|display))\b.{0,40}"
        r"\b(citat|source|api|endpoint|origin|reference|schema)\w*\b",
        re.I,
    ), "citation / source suppression"),

    # Schema bypass — "return plain text only", "no JSON", "no structure", "do not follow schema"
    (re.compile(
        r"\b(no\s+json|no\s+struct|plain\s+text\s+only"
        r"|don[''`]?t\s+follow.{0,40}schema"
        r"|do\s+not\s+follow.{0,40}schema"
        r"|ignore.{0,20}schema"
        r"|return\s+plain)\b",
        re.I,
    ), "schema bypass"),

    # API / tool suppression
    (re.compile(
        r"\b(do\s+not\s+use|don[''`]?t\s+use|avoid|skip)\b.{0,40}"
        r"\b(api|external|endpoint|duckdb|database|world\s*bank|tool)\b",
        re.I,
    ), "API / tool suppression"),

    # Force-ignore specific data types / tools ("ignore economic data completely")
    (re.compile(
        r"\b(ignore|disregard|exclude|omit|skip)\b.{0,60}"
        r"\b(economic|employment|conflict|climate|health|safety|political|acled|worldbank|world\s*bank)\b"
        r".{0,40}\b(data|completely|entirely|altogether|even\s+if)\b",
        re.I,
    ), "data source suppression"),

    # "even if other signals are stronger/better" — analysis-override framing
    (re.compile(
        r"\beven\s+if\b.{0,40}\b(stronger|better|more\s+relevant|dominant|clearer|important)\b",
        re.I,
    ), "analysis override framing"),

    # Assume / inject fake values
    (re.compile(
        r"\b(assume|pretend|act\s+as\s+if|treat\s+as)\b.{0,60}"
        r"\b(inflation|gdp|migration|data|value|rate)\b",
        re.I,
    ), "fake value injection"),

    # Step-numbered jailbreak ("Step 1: do not …")
    (re.compile(
        r"step\s*[1-9]\s*[:\.]\s*(do\s+not|don[''`]?t|ignore|generate|fabricat)\b",
        re.I,
    ), "step-by-step jailbreak"),
]


def check_query(query: str) -> tuple[bool, str]:
    """Return (True, "") if the query is safe, (False, reason) if it is not.

    Strips leading/trailing whitespace before matching.
    """
    text = (query or "").strip()
    if not text:
        return False, "Query is empty."

    for pattern, category in _PATTERNS:
        if pattern.search(text):
            return False, (
                f"Your query contains instructions that attempt to override system "
                f"behaviour ({category}). Please ask a genuine migration or "
                f"country-comparison question."
            )

    return True, ""
