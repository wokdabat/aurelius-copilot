import re

def extract_kpis_from_chunks(chunks):
    text = " ".join([c["text"] for c in chunks])

    def find(pattern):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    return {
        "revenue": find(r"Total revenue.*?([\d\.]+) billion"),
        "gross_margin": find(r"Gross margin.*?(\d+)%"),
        "operating_income": find(r"Operating income.*?([\d\.]+) billion"),
        "rd_ratio": find(r"R&D Investment Ratio.*?(\d+\.?\d*)%"),
        "primary_metric": find(r"CCY[:\s]+([\d\.]+)")
    }