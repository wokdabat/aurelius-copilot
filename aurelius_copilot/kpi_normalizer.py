# aurelius_copilot/kpi_normalizer.py

import re

# ---------------------------------------------------------
# 1. Canonical KPI Synonym Map (all → extractor’s real keys)
# ---------------------------------------------------------
KPI_SYNONYMS = {
    # R&D → rd_ratio (your extractor’s canonical key)
    "rd_ratio": [
        "r&d investment ratio",
        "r&d ratio",
        "r&d intensity",
        "r&d expenditure ratio",
        "r&d as % of revenue",
        "r and d intensity",
        "research and development intensity",
        "research and development ratio",
        "r&d",
        "r and d",
        "rnd",
        "r d",
        "rd",
        "research and development",
        "research & development",
        "r&d spend",
        "r&d expenses",
        "r&d costs",
        "r&d spending",
        "r&d expenditure",
        "r&d percentage",
    ],

    # CCY → primary_metric (your extractor’s key)
    "primary_metric": [
        "ccy",
        "cognitive compute yield",
        "compute yield",
        "neural compute yield",
        "constant currency",
    ],

    # Hydrogen Conversion Efficiency → hce
    "hce": [
        "hce",
        "hydrogen conversion efficiency",
        "conversion efficiency",
    ],

    # Production Efficiency → production_efficiency
    "production_efficiency": [
        "production efficiency",
        "operational efficiency",
        "manufacturing efficiency",
        "production efficiency index",
    ],

    # Adoption Rate → adoption_rate
    "adoption_rate": [
        "adoption rate",
        "customer adoption rate",
        "market adoption rate",
        "adoption percentage",
    ],

    # Revenue Growth → revenue_growth
    "revenue_growth": [
        "revenue growth",
        "yoy revenue growth",
        "year-over-year revenue growth",
        "growth rate",
    ],

    # Revenue → revenue
    "revenue": [
        "revenue",
        "total revenue",
        "sales",
        "top line",
    ],

    # Gross Margin → gross_margin
    "gross_margin": [
        "gross margin",
        "gm",
        "gross profit margin",
        "margin",
    ],

    # Operating Income → operating_income
    "operating_income": [
        "operating income",
        "operating profit",
        "op income",
        "op profit",
    ],
}
# ---------------------------------------------------------
# 2. Normalize a user query to canonical KPI key
# ---------------------------------------------------------
def normalize_metric_name(name: str) -> str:
    if not name:
        return None

    n = name.lower().strip()
    n = re.sub(r"[^a-z0-9&% ]+", "", n)
    n = re.sub(r"\s+", " ", n)

    # Direct synonym match
    for canonical, synonyms in KPI_SYNONYMS.items():
        if n == canonical:
            return canonical
        if n in synonyms:
            return canonical

    # Fuzzy contains match (e.g., “r&d intensity metric”)
    for canonical, synonyms in KPI_SYNONYMS.items():
        if any(s in n for s in synonyms):
            return canonical

    return None


# ---------------------------------------------------------
# 3. Extract numeric values from retrieved text
# ---------------------------------------------------------
def extract_metric_value(text: str, canonical_metric: str) -> float | None:
    text_l = text.lower()

    # If canonical_metric is None, no extraction should occur
    if not canonical_metric:
        return None

    # Always get a list of synonyms; fallback is an empty list, not [None]
    synonyms = KPI_SYNONYMS.get(canonical_metric, [])

    for syn in synonyms:
        if not syn:
            continue  # skip None or empty synonyms

        syn_l = syn.lower()
        if syn_l in text_l:
            # % value
            match = re.search(r"(\d+(\.\d+)?)\s*%", text_l)
            if match:
                return float(match.group(1)) / 100.0

            # plain number
            match = re.search(r"(\d+(\.\d+)?)", text_l)
            if match:
                return float(match.group(1))

    return None