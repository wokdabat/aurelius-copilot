# frontend/app.py
import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.cache_data.clear()

st.set_page_config(
    page_title="Aurelius Analyst Copilot",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Aurelius Corporate Exchange — Analyst Dashboard")

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("Analysis Controls")

company = st.sidebar.selectbox(
    "Select Company",
    [
        "NeuroFlux Systems",
        "VerdantWave Energy",
        "Genova BioWorks",
        "Atlas Robotics Group",
        "SkyLink Aeronautics"
    ]
)

query = st.sidebar.text_area(
    "Enter your analysis request",
    "Analyze key performance trends and risks."
)

run_button = st.sidebar.button("Run Analysis")

# -----------------------------
# Main Dashboard
# -----------------------------
if run_button:
    with st.spinner("Running multi-agent analysis..."):
        response = requests.post(
            "http://127.0.0.1:8000/analyze",
            json={"company": company, "query": query}
        )
        result = response.json()
        
    # Extract fields from backend
    #canonical_metric = result["canonical_metric"]
    kpis = result["kpis"]
    kpi_evidence = result["kpi_evidence"]
    insight = result["narrative"]
    chunks = result["retrieved_chunks"]

    # -----------------------------
    # Final Insight
    # -----------------------------
    st.markdown(
        f"""
    ### Final Analyst Insight

    {insight}
    """
    )
'''
    # -----------------------------
    # KPI Card (single canonical metric)
    # -----------------------------
    st.subheader("Key Performance Indicator")

    # Evidence for KPI
    st.markdown("### Evidence")
    if kpi_evidence:
        for e in kpi_evidence:
            st.write(f"- {e}")
    else:
        st.write("No direct evidence found in retrieved filings.")

    st.markdown("---")

    # -----------------------------
    # Trend Charts (placeholder synthetic data)
    # -----------------------------
    st.subheader("Performance Trends")

    df = pd.DataFrame({
        "Year": [2021, 2022, 2023, 2024, 2025],
        "Revenue": [7.2, 8.1, 9.4, 10.7, 12.8],
        "GrossMargin": [45, 47, 48, 49, 54]
    })

    fig_rev = px.line(df, x="Year", y="Revenue", title="Revenue Trend (B credits)")
    fig_margin = px.line(df, x="Year", y="GrossMargin", title="Gross Margin Trend (%)")

    colA, colB = st.columns(2)
    colA.plotly_chart(fig_rev, use_container_width=True)
    colB.plotly_chart(fig_margin, use_container_width=True)

    st.markdown("---")

    # -----------------------------
    # Retrieved Filing Context
    # -----------------------------
    st.subheader("Retrieved Filing Context")

    with st.expander("Retrieved Filing Chunks (click to expand)"):
        for chunk in chunks:
            st.write(chunk["content"])

    st.markdown("---")

    st.caption("Powered by CrewAI multi-agent RAG pipeline")
'''