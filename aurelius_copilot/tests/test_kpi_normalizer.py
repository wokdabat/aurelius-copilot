import pytest
from aurelius_copilot.kpi_normalizer import normalize_metric_name, extract_metric_value, KPI_SYNONYMS

# ---------------------------------------------------------
# 1. Test R&D normalization
# ---------------------------------------------------------

RD_SYNONYMS = KPI_SYNONYMS["rd_ratio"]

@pytest.mark.parametrize("syn", RD_SYNONYMS)
def test_rd_synonyms_map_to_rd_ratio(syn):
    assert normalize_metric_name(syn) == "rd_ratio"

def test_rd_fuzzy_match():
    assert normalize_metric_name("r&d intensity metric") == "rd_ratio"
    assert normalize_metric_name("rd metric") == "rd_ratio"
    assert normalize_metric_name("research and development spend details") == "rd_ratio"

# ---------------------------------------------------------
# 2. Test CCY normalization
# ---------------------------------------------------------

@pytest.mark.parametrize("syn", KPI_SYNONYMS["primary_metric"])
def test_ccy_synonyms_map_to_primary_metric(syn):
    assert normalize_metric_name(syn) == "primary_metric"

def test_ccy_fuzzy_match():
    assert normalize_metric_name("neural compute yield metric") == "primary_metric"

# ---------------------------------------------------------
# 3. Test revenue, margin, operating income
# ---------------------------------------------------------

@pytest.mark.parametrize("syn", KPI_SYNONYMS["revenue"])
def test_revenue_synonyms(syn):
    assert normalize_metric_name(syn) == "revenue"

@pytest.mark.parametrize("syn", KPI_SYNONYMS["gross_margin"])
def test_gross_margin_synonyms(syn):
    assert normalize_metric_name(syn) == "gross_margin"

@pytest.mark.parametrize("syn", KPI_SYNONYMS["operating_income"])
def test_operating_income_synonyms(syn):
    assert normalize_metric_name(syn) == "operating_income"

# ---------------------------------------------------------
# 4. Test numeric extraction
# ---------------------------------------------------------

def test_extract_percentage_value():
    text = "The R&D Investment Ratio increased to 12.5% this year."
    assert extract_metric_value(text, "rd_ratio") == 0.125

def test_extract_plain_number():
    text = "The R&D ratio reached 15 this quarter."
    assert extract_metric_value(text, "rd_ratio") == 15.0

def test_extract_none_when_no_match():
    text = "No relevant metric here."
    assert extract_metric_value(text, "rd_ratio") is None

# ---------------------------------------------------------
# 5. Regression tests: ensure canonical keys never drift
# ---------------------------------------------------------

def test_canonical_keys_match_extractor():
    expected_keys = {
        "rd_ratio",
        "primary_metric",
        "hce",
        "production_efficiency",
        "adoption_rate",
        "revenue_growth",
        "revenue",
        "gross_margin",
        "operating_income",
    }
    assert set(KPI_SYNONYMS.keys()) == expected_keys