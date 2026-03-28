# dashboard.py
import streamlit as st
import plotly.express as px
import requests
import pandas as pd
import textwrap
import altair as alt
from fpdf import FPDF

API_URL = "http://127.0.0.1:8000"

st.markdown(
    """
    <style>
      /* Hide any <code>None</code> or code block containing exactly "None" */
      code {
        display: none !important;
      }

      p > code:only-child,
      div[data-testid="stMarkdownContainer"] > p > code {
        display: none !important;
      }

      /* If it's in a small/emotion container */
      .st-emotion-cache-znj1k1:where(code) {
        display: none !important;
      }

      /* Optional: hide only if text is exactly "None" (more precise) */
      code::after {
        content: "";
      }
    </style>
    """,
    unsafe_allow_html=True
)
st.set_page_config(page_title="Aurelius Analyst Dashboard", layout="wide")

st.title("Aurelius Analyst Dashboard")
st.write("Run single‑company Financial analysis using your CrewAI backend.")

def force_break_long_tokens(text, max_len=30):
    safe_tokens = []
    for token in text.split(" "):
        if len(token) > max_len:
            chunks = [token[i:i+max_len] for i in range(0, len(token), max_len)]
            safe_tokens.extend(chunks)
        else:
            safe_tokens.append(token)
    return " ".join(safe_tokens)

def clean_insight_text(text: str) -> str:
    if not text:
        return "No insight available."

    # Fix spacing around numbers and punctuation
    import re
    text = re.sub(r'(\d)\s*\.\s*(\d)', r'\1.\2', text)          # 10 . 1 → 10.1
    text = re.sub(r'(\d)\s*([,%])', r'\1\2', text)              # 16 % → 16%
    text = re.sub(r'\s+', ' ', text)                            # collapse extra spaces
    text = text.replace(" . ", ". ").replace(" , ", ", ")

    # Remove trailing incomplete words/sentences (heuristic)
    text = re.sub(r'\b\w{1,3}$', '', text.strip())              # cut off very short ending fragments
    if not text.endswith(('.', '!', '?')):
        text = text.rstrip(' ,;') + '...'                       # graceful ellipsis if incomplete

    # Capitalize sentence starts (basic)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    cleaned_sentences = []
    for s in sentences:
        s = s.strip()
        if s:
            cleaned_sentences.append(s[0].upper() + s[1:])
    text = ' '.join(cleaned_sentences)

    return text.strip()

# ────────────────────────────────────────────────
# HELPER: Clean & format KPI values nicely
# ────────────────────────────────────────────────
def format_kpi_value(value, kpi_name: str) -> str:
    """Clean KPI value: handle numbers, percentages, remove trailing decimals/zeros"""
    if value is None or value == "":
        return "—"

    val_str = str(value).strip()

    # Handle dict-like strings (if still appearing)
    if val_str.startswith("{") and val_str.endswith("}"):
        try:
            cleaned = (
                val_str.replace("'", '"')
                .replace("Prior Year", "prior")
                .replace("Current Year", "current")
                .replace("Percentage of Total Revenue", "percentage")
            )
            d = eval(cleaned, {"__builtins__": {}})
            # Pick most relevant value (customize order if needed)
            for key in ["current", "Current Year", "percentage", "Percentage", "value"]:
                if key in d:
                    val_str = str(d[key])
                    break
            else:
                val_str = ", ".join(f"{k}: {v}" for k, v in d.items() if v is not None)
        except Exception:
            val_str = val_str.replace("{", "").replace("}", "").replace("'", "").strip()

    # Try numeric conversion
    try:
        num = float(val_str)
    except (ValueError, TypeError):
        return val_str  # non-numeric → return as-is

    # Detect percentage KPIs (expanded list)
    name_lower = kpi_name.lower()
    is_percentage = any(
        word in name_lower
        for word in [
            "margin", "rate", "percentage", "percent", "ratio", "yield",
            "utilization", "adoption", "growth", "%", "roi", "utilisation"
        ]
    )

    if is_percentage:
        # For percentages: show as whole number if possible, max 2 decimals otherwise
        if num.is_integer() or abs(num - round(num, 0)) < 1e-8:
            return f"{int(round(num))}%"
        else:
            formatted = f"{num:.2f}%".rstrip("0").rstrip(".")
            return formatted if formatted.endswith("%") else formatted + "%"

    # Normal numbers
    if num.is_integer():
        return f"{int(num):,}"           # 10100 → 10,100
    else:
        # Show clean float without trailing zeros
        return f"{num:g}"
    
def humanize_kpi_name(name: str) -> str:
    """Convert snake_case to Title Case: total_revenue → Total Revenue"""
    return name.replace("_", " ").title()

# ---------------------------------------------------------
# PDF GENERATOR — includes KPIs + Evidence
# ---------------------------------------------------------

def generate_pdf(narrative: str, kpis: dict = None, kpi_evidence: list = None, chart_png_bytes=None) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=18)

    pdf.add_font("DejaVu", "", "aurelius_copilot/fonts/DejaVuSans.ttf", uni=True)
    pdf.add_font("DejaVu", "B", "aurelius_copilot/fonts/DejaVuSans-Bold.ttf", uni=True)

    pdf.set_font("DejaVu", size=12)

    # Title
    pdf.set_font("DejaVu", 'B', 20)
    pdf.cell(0, 15, "Aurelius Analyst Report", ln=True, align="C")
    pdf.ln(20)

    # Narrative / Insight
    pdf.set_font("DejaVu", 'B', 14)
    pdf.cell(0, 12, "Final Analyst Insight", ln=True)
    pdf.ln(12)
    pdf.set_font("DejaVu", '', 10)
    pdf.set_text_color(30, 30, 30)

    body_width = pdf.w - 2 * pdf.l_margin
    if body_width < 10:
        body_width = 100

    paragraphs = [p.strip() for p in narrative.split("\n\n") if p.strip()] or [narrative.strip()]

    for para in paragraphs:
        para = clean_insight_text(para)
        cleaned = force_break_long_tokens(para, max_len=45)
        wrapped = textwrap.wrap(cleaned, width=1400, break_long_words=True, break_on_hyphens=True)
        for line in wrapped:
            pdf.multi_cell(body_width, 8.5, line, align="J")
        pdf.ln(6)

    pdf.ln(15)

    # KPIs table
    if kpis and len(kpis) > 0:
        pdf.set_font("DejaVu", 'B', 14)
        pdf.cell(0, 12, "Extracted KPIs", ln=True)
        pdf.ln(8)

        pdf.set_font("DejaVu", 'B', 10)
        pdf.cell(100, 10, "KPI", border=1, align="L")
        pdf.cell(60, 10, "Value", border=1, align="C", ln=True)

        pdf.set_font("DejaVu", '', 10)
        for raw_name, raw_value in kpis.items():
            clean_name = raw_name.replace("_", " ").title()
            display = format_kpi_value(raw_value, raw_name)  # reuse your formatter

            pdf.cell(100, 10, clean_name[:90] + ("..." if len(clean_name) > 90 else ""), border=1, align="L")
            pdf.cell(60, 10, display, border=1, align="R", ln=True)

        pdf.ln(20)  # more space after table

    # Chart section — only if bytes provided
    if chart_png_bytes:
        pdf.add_page()  # dedicate a full page to avoid overlap/cutoff
        pdf.set_font("DejaVu", 'B', 14)
        pdf.cell(0, 10, "Key Metrics Breakdown", ln=True, align="C")
        pdf.ln(10)

        try:
            pdf.image(
                chart_png_bytes,
                x=10,                   # left margin
                y=pdf.get_y(),
                w=190,                  # almost full width
                h=0,                    # auto height to preserve aspect ratio
                type='PNG'
            )
            pdf.ln(5)  # small space after image
            print("PDF: Chart image embedded successfully")
        except Exception as img_err:
            print(f"PDF: Chart embedding failed: {img_err}")
            pdf.set_font("DejaVu", '', 11)
            pdf.multi_cell(0, 8, "[Chart image could not be included]", align="C")
    else:
        print("PDF: No chart bytes provided — skipping chart page")

    # KPI Evidence
    if kpi_evidence and len(kpi_evidence) > 0:
        # Only new page if near bottom or after chart
        if pdf.get_y() > 180 or chart_png_bytes:
            pdf.add_page()
        pdf.set_font("DejaVu", 'B', 14)
        pdf.cell(0, 12, "KPI Evidence", ln=True)
        pdf.ln(8)
        pdf.set_font("DejaVu", '', 10)
        for item in kpi_evidence:
            cleaned = item.lstrip("-• ").strip()
            pdf.multi_cell(0, 8, f"• {cleaned}")
            pdf.ln(4)

    pdf_raw = pdf.output(dest="S")
    if isinstance(pdf_raw, bytearray):
        pdf_raw = bytes(pdf_raw)
    return pdf_raw

# ---------------------------------------------------------
# MODE SELECTOR
# ---------------------------------------------------------
mode = st.radio(
    "",
    "Single Company",
    horizontal=True
)

# ---------------------------------------------------------
# INPUTS
# ---------------------------------------------------------
st.markdown(
    "<h2 style='font-size: 30px; font-weight: 700; margin-bottom: 0;'>Company Name</h2>",
    unsafe_allow_html=True
)
if mode == "Single Company":
    company = st.text_input("Please choose from Atlas, Neuroflux, Genova, Skylink, and VerdantWave",
    help="Please chose from Atlas, Neuroflux, Genova, Skylink, and VerdantWave.", 
    placeholder="e.g., Atlas, Genova")
else:
    companies_raw = st.text_input(
        "Companies (comma‑separated)",
        placeholder="e.g., Atlas, Genova, VerdantWave"
    )
st.markdown(
    "<h2 style='font-size: 30px; font-weight: 700; margin-bottom: 0;'>Analysis Query</h2>",
    unsafe_allow_html=True
)

st.markdown(
    """
    **Below are some sample queries you can use:**

    - Describe the innovation strategy and operational efficiency.
    - What are the company’s main growth drivers right now?
    - How is the company investing in innovation or R&D?
    - Describe the company’s competitive position in its industry.
    - Summarize the company’s recent financial performance.
    """
)

query = st.text_area(
    "",
    placeholder="e.g., Compare their innovation strategy and operational efficiency."
)

run_button = st.button("Run Analysis", type="primary")


# ---------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------
if run_button:
    if not query.strip():
        st.error("Please enter a query.")
        st.stop()

    if mode == "Single Company":
        if not company.strip():
            st.error("Please enter a company name.")
            st.stop()

        payload = {
            "company": company.strip(),
            "query": query.strip()
        }

        endpoint = f"{API_URL}/analyze"

    else:
        if not companies_raw.strip():
            st.error("Please enter at least one company.")
            st.stop()

        companies = [c.strip() for c in companies_raw.split(",") if c.strip()]

        payload = {
            "companies": companies,
            "query": query.strip()
        }

        endpoint = f"{API_URL}/compare"

    # -----------------------------------------------------
    # CALL BACKEND
    # -----------------------------------------------------
    with st.spinner("Running analysis..."):
        try:
            response = requests.post(endpoint, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.HTTPError as http_err:
            st.error(f"Backend error ({response.status_code}): {response.text[:300]}")
            st.stop()
        except requests.exceptions.Timeout:
            st.error("Backend timed out. Please try again.")
            st.stop()
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to backend. Is the server running?")
            st.stop()
        except ValueError:
            st.error(f"Backend response is not valid JSON:\n{response.text[:300]}")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.stop()

    # -----------------------------------------------------
    # DISPLAY RESULTS
    # -----------------------------------------------------
    st.subheader("Final Analyst Insight")

    insight = result.get("narrative", "")
    
    if insight:
        insight = clean_insight_text(insight)
        st.markdown(insight)        
    else:
        st.warning("No narrative insight returned. Please use a valid query.")
        with st.expander("Raw backend response (debug)"):
            st.json(result)
    
    kpis = result.get("kpis", {})
    kpi_evidence = result.get("kpi_evidence", [])
   
    # -----------------------------------------------------
    # KPI DISPLAY (Single Company)
    # -----------------------------------------------------
    if mode == "Single Company":
        #canonical_metric = result.get("canonical_metric")
        kpis = result.get("kpis", {})
        kpi_evidence = result.get("kpi_evidence", [])

        if kpis:
            st.subheader("All Extracted KPIs")

            rows = []
            for raw_name, raw_value in kpis.items():
                clean_name = humanize_kpi_name(raw_name)
                display_value = format_kpi_value(raw_value, raw_name)  # pass raw_name for % detection
                rows.append({"KPI": clean_name, "Value": display_value})

            df = pd.DataFrame(rows).set_index("KPI")

            # Optional: make values right-aligned (via styler)
            styled = df.style.set_properties(
                **{'text-align': 'right'}, subset=['Value']
            )

            st.dataframe(styled, use_container_width=True)
            # or keep st.table(df) if you prefer the simpler look

            # ────────────────────────────────────────────────
            # PIE CHART SECTION
            # ────────────────────────────────────────────────
            with st.container():
                st.subheader("Key Metrics Breakdown")

                chart_data = []
                for raw_name, raw_value in kpis.items():
                    try:
                        val_str = str(raw_value).replace("%", "").replace(" billion credits", "").replace(" million credits", "").replace(",", "").strip()
                        num = float(val_str)
                        if 0 < num <= 100 or any(word in raw_name.lower() for word in ["ratio", "rate", "adoption", "utilization", "margin", "percent"]):
                            chart_data.append({
                                "KPI": humanize_kpi_name(raw_name),
                                "Value": num
                            })
                    except (ValueError, TypeError):
                        pass

                fig = None

                if chart_data:
                    df_chart = pd.DataFrame(chart_data)

                    fig = px.pie(
                        df_chart,
                        values='Value',
                        names='KPI',
                        title='Key Percentage / Ratio Metrics',
                        hole=0.3,
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )

                    fig.update_traces(
                        textposition='outside',           # ← move labels outside to reduce crowding
                        textinfo='percent+label',
                        insidetextorientation='radial',
                        textfont_size=12
                    )

                    fig.update_layout(
                        showlegend=False,
                        legend_title_text='KPIs',
                        height=480,                       # slightly smaller to fit better
                        margin=dict(t=60, b=80, l=20, r=20),  # extra bottom margin for legend
                        legend=dict(
                            orientation="h",              # horizontal legend → less vertical intrusion
                            yanchor="bottom",
                            y=-0.3,                       # push legend below chart
                            xanchor="center",
                            x=0.5,
                            bgcolor="rgba(0,0,0,0.8)",  # optional: slight background for readability
                            bordercolor="gray",
                            borderwidth=1
                        )
                    )

                    st.plotly_chart(fig, use_container_width=True)

                else:
                    st.info("No percentage or ratio KPIs suitable for a pie chart.")

            # ── Explicit spacer to prevent overlap with next section ──
            st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
            # or use st.markdown("---") if you want a visible divider line

            # ────────────────────────────────────────────────
            # KPI EVIDENCE SECTION
            # ────────────────────────────────────────────────
            if kpi_evidence:
                st.subheader("KPI Evidence")
                for evidence_item in kpi_evidence:
                    cleaned = evidence_item.lstrip("-• ").strip()
                    st.markdown(f"- {cleaned}")

            # After creating fig in your pie chart code
            import io

            pdf_bytes = None  # default fallback

            if fig is not None:
                try:
                    img_bytes = io.BytesIO()
                    fig.write_image(img_bytes, format="png", engine="kaleido", width=800, height=600)
                    img_bytes.seek(0)
                    print("Chart PNG size:", len(img_bytes.getvalue()))

                    pdf_bytes = generate_pdf(
                        insight,
                        kpis,
                        kpi_evidence,
                        chart_png_bytes=img_bytes
                    )
                except Exception as e:
                    print(f"Chart export failed: {e}")
                    pdf_bytes = generate_pdf(insight, kpis, kpi_evidence)  # fallback without chart
            else:
                pdf_bytes = generate_pdf(insight, kpis, kpi_evidence)  # no chart was made

      
            st.download_button(
                    label="Download Report as PDF",
                    data=pdf_bytes,
                    file_name="Analysis_Report.pdf",
                    mime="application/pdf"
                )      