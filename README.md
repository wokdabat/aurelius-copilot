# Aurelius Copilot

**Intelligent Financial Analysis System**  
*Transform raw company data into structured insights, KPI dashboards, risk assessments, and professional PDF reports — powered by multi-agent AI.*

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![CrewAI](https://img.shields.io/badge/CrewAI-000000?style=for-the-badge&logo=crewai&logoColor=white)

---

## Motivation (My true motivation was to provide a faster way to search for information on my company's wiki pages. Often time at work I will hear I cannot find anything on this wiki page, well why not tie in the wiki and AI, search and instantly find what you were looking for.)

Financial analysis remains slow, inconsistent, and highly manual. Analysts spend hours gathering data from filings, interpreting signals, identifying risks, synthesizing narratives, and producing polished reports.

**Aurelius Copilot** solves this by automating the entire workflow with a modular, multi-agent AI system. It delivers consistent, high-quality insights in minutes — combining retrieval-augmented generation (RAG), structured reasoning, risk enforcement, and professional reporting.

Built for clarity, modularity, and reproducibility, it empowers financial analysts, investors, and researchers to focus on high-value interpretation rather than tedious data wrangling.

## Features

- **Multi-Agent Reasoning** — Powered by CrewAI with specialized agents for research, analysis, risk assessment, and synthesis
- **Hybrid Retrieval** — Combines dense embeddings (semantic understanding) with BM25 keyword search (precision on jargon and rare terms) for superior evidence quality
- **Risk Enforcement Logic** — Ensures risk-focused queries always surface explicit risk insights
- **Interactive Dashboard** — Clean Streamlit UI with colored KPI tables, visualizations, and evidence grouping
- **Professional PDF Reports** — One-click generation of multi-page reports with embedded charts and formatted narrative
- **Query Classification** — Automatically routes queries (risk, growth, competitive, general) to the optimal agent workflow
- **Modular & Extensible** — Easy to add new agents, tools, or data sources

## Architecture

### System Sequence Diagram (End-to-End Flow)

```mermaid
sequenceDiagram
    participant User
    participant Streamlit
    participant FastAPI
    participant Orchestrator
    participant CrewAI
    participant Retrieval
    participant VectorStore
    participant PDF

    User->>Streamlit: Submit company name and query
    Streamlit->>FastAPI: POST /analyze request
    FastAPI->>Orchestrator: Forward request + classification
    Orchestrator->>CrewAI: Initialize agents and tasks

    CrewAI->>Retrieval: Request relevant evidence
    Retrieval->>VectorStore: Dense embeddings search
    Retrieval->>VectorStore: BM25 keyword search
    VectorStore-->>Retrieval: Return top chunks
    Retrieval-->>CrewAI: Merged & ranked evidence

    loop Multi-Agent Reasoning
        CrewAI->>CrewAI: Research Agent
        CrewAI->>CrewAI: Analysis Agent extracts KPIs
        CrewAI->>CrewAI: Risk Agent evaluates risks
        CrewAI->>CrewAI: Synthesis Agent builds narrative
    end

    CrewAI-->>Orchestrator: Final structured output
    Orchestrator->>Orchestrator: Apply Risk Enforcement Logic
    Orchestrator-->>FastAPI: Return JSON response
    FastAPI-->>Streamlit: Display dashboard

    alt User requests PDF
        Streamlit->>PDF: Generate professional PDF report
        PDF-->>Streamlit: Provide download link
    end

    Streamlit-->>User: Show KPI tables, charts and insights

## Tech Stack

Framework: FastAPI (backend API) + Streamlit (frontend dashboard)
Multi-Agent Engine: CrewAI
Retrieval:
- Dense Embeddings: sentence-transformers (e.g., all-mpnet-base-v2 or finance-tuned variant) + FAISS
- Keyword Search: rank_bm25 (BM25Okapi)
- Hybrid Fusion: Combined ranking with deduplication
PDF Generation: FPDF2 with custom text wrapping and image embedding
Data Handling: Pydantic for structured outputs, Pandas for KPI tables
Visualization: Matplotlib / Plotly (via Streamlit)

Installation & Quick Start
Prerequisites

Python 3.10+
Basic familiarity with virtual environments

Steps

Clone the repositoryBashgit clone https://github.com/yourusername/aurelius-copilot.git
cd aurelius-copilot
Create and activate virtual environmentBashpython -m venv venv (I used uv, uv init .)
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
Install dependenciesBashpip install -r requirements.txt
Prepare your data
Place company filings or documents in the data/ folder
Run the ingestion script to build embeddings + BM25 index (see ingest.py)

Run the applicationBash# Start FastAPI backend
uv run --project . python -m aurelius_copilot.api.server

# In a new terminal, start Streamlit frontend
uv run streamlit run .\dashboard.py

Open http://localhost:8501 in your browser to start analyzing companies.
Example Queries & Outputs
Example 1: Risk-Focused Query
text"Analyze supply chain and competitive risks"
Expected Output:

Final Insight
Colored KPI table with risk status
Dedicated risk section with enforced content
Evidence citations from filings
Downloadable PDF report

Example 2: Growth Analysis
text"Evaluate revenue growth drivers and opportunities for NVIDIA"
Screenshots (add these to your repo):
<img src="screenshots/kpi_table.png" alt="KPI Dashboard">
<img src="screenshots/pdf_preview.png" alt="PDF Report Preview">
(Replace with actual screenshots once generated)
Limitations & Future Improvements
Current Limitations

Performance depends on quality and recency of ingested documents
Hybrid retrieval requires careful tuning of fusion weights
PDF generation currently uses basic styling (can be enhanced with ReportLab or WeasyPrint)
Single-company analysis (no built-in peer benchmarking yet)

##Planned Enhancements

Industry benchmarking across peer companies
Time-series forecasting for key metrics
Live market data integration (via APIs)
Batch processing for multiple companies
User authentication and saved report history
Advanced reranking (cross-encoder) and Reciprocal Rank Fusion (RRF) improvements

##References

CrewAI Documentation: https://docs.crewai.com
Hybrid Retrieval Best Practices (BM25 + Embeddings)
NVIDIA / BlackRock research on Hybrid RAG for financial documents
Sentence-Transformers library and finance-domain embedding models
rank_bm25 for lexical precision in technical domains


Made with ❤️ for financial analysts who value speed without sacrificing depth.
Contributions, issues, and feature requests are welcome!
Star the repo if you find it useful ⭐