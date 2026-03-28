# aurelius_copilot/api/routes.py
from fastapi import APIRouter
from pydantic import BaseModel
from aurelius_copilot.agents.orchestrator import(
    run_financial_analysis
)
from aurelius_copilot.tools.hybrid_retrieval_tool import hybrid_retrieve_tool

router = APIRouter()

# -----------------------------
# REQUEST MODELS
# -----------------------------
class AnalyzeRequest(BaseModel):
    company: str
    query: str


class CompareRequest(BaseModel):
    companies: list[str]
    query: str


# -----------------------------
# SINGLE-COMPANY ANALYSIS
# -----------------------------
@router.post("/analyze")
def analyze(payload: AnalyzeRequest):
    company = payload.company
    query = payload.query

    result = run_financial_analysis(
        company=company,
        query=query        
    )

    final_output = result.get("final_output", "")

    return {
        "company": company,
        "query": query,
        "narrative": final_output,
        #"canonical_metric": result.get("canonical_metric"),
        "kpis": result.get("kpis", {}),
        "kpi_evidence": result.get("kpi_evidence", []),
        "raw_kpis": result.get("raw_kpis", {}),
        "raw_kpi_evidence": result.get("raw_kpi_evidence", {}),
        "retrieved_chunks": result.get("tasks_output", [])
    }