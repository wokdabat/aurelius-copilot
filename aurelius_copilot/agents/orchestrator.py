import json
from crewai import Crew, Process, Task
from .agents import (
    retrieval_agent,
    analysis_agent,
    generation_agent,
    critic_agent
)
from aurelius_copilot.kpi_normalizer import normalize_metric_name, extract_metric_value
from aurelius_copilot.kpi_normalizer import KPI_SYNONYMS

def extract_metric_name(query: str) -> str:
    q = query.lower()

    if "compare" in q:
        after = q.split("compare", 1)[1].strip()

        for keyword in ["across", "between", "for", "among"]:
            if keyword in after:
                metric = after.split(keyword, 1)[0].strip()
                break
        else:
            metric = after

        # Normalize deterministically
        metric = metric.strip()
        metric = metric.replace("the ", "", 1)  # remove leading "the "
        metric = metric.replace("a ", "", 1)    # remove leading "a "
        metric = metric.replace("an ", "", 1)   # remove leading "an "

        return metric.strip()

    return query.strip()
# ---------------------------------------------------------
# KPI EXTRACTION
# ---------------------------------------------------------
def extract_kpis_from_chunks(query: str, retrieved_chunks: list[str]):
    kpi_values = {metric: None for metric in KPI_SYNONYMS.keys()}
    evidence = {metric: [] for metric in KPI_SYNONYMS.keys()}

    for chunk in retrieved_chunks:
        for metric in KPI_SYNONYMS.keys():
            val = extract_metric_value(chunk, metric)
            if val is not None:
                kpi_values[metric] = val
                evidence[metric].append(chunk.strip())

    return kpi_values, evidence
# ---------------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------------
def normalize_company_name(name: str) -> str:
    name = name.lower().strip()

    mapping = {
        "atlas robotics group": "Atlas",
        "atlas": "Atlas",
        "neuroflux systems": "Neuroflux",
        "neuroflux": "Neuroflux",
        "genova bioworks": "Genova",
        "genova": "Genova",
        "skylink aeronautics": "Skylink",
        "skylink": "Skylink",
        "verdantwave energy": "Verdantwave",
        "verdantwave": "Verdantwave",
    }

    return mapping.get(name, name.title())
# ---------------------------------------------------------
# SINGLE-COMPANY FLOW
# ---------------------------------------------------------
def run_single_company_flow(company: str, query: str):
    query = query.strip()
    if not query or len(query) < 10 or "joke" in query.lower() or not any(word in query.lower() for word in ["strategy", "growth", "financial", "innovation", "r&d", "competitive", "performance", "drivers", "position", "metric", "rate", "investment"]):
        return {
            "narrative": "This doesn't seem like a valid analysis question. Please ask something specific about the company's strategy, innovation/R&D, growth drivers, competitive position, financial performance, or a particular metric.",
            "kpis": {},
            "kpi_evidence": [],
            "retrieved_chunks": [],
            "canonical_metric": None
        }
    normalized_company = normalize_company_name(company)

    # 1. Retrieval
    retrieval_task = Task(
        description=f"Retrieve relevant chunks for {normalized_company}.",
        agent=retrieval_agent,
        expected_output="retrieved_chunks",
        output_key="retrieved_chunks",
        inputs={"query": query, "company": normalized_company}
    )
    # 1.5 KPI Extraction (NEW)
    kpi_extraction_task = Task(
        description=(
            "Extract ALL KPIs from retrieved_chunks. "
            "Return STRICT JSON ONLY with exactly two keys: "
            "kpis (object) and evidence (object). "
            "kpis MUST contain every canonical KPI from the KPI_SYNONYMS list. "
            "evidence MUST map each KPI to the list of chunks where it was found. "
            "Do NOT summarize. Do NOT infer. Only extract numeric KPI values."
        ),
        agent=analysis_agent,
        expected_output="kpi_json",
        output_key="kpi_json",
        inputs={
            "retrieved_chunks": "{{ retrieved_chunks }}"
        }
    )
    # 2. Analysis (now receives extracted KPIs + evidence)
    analysis_task = Task(
        description=(
            "Analyze the retrieved_chunks and extracted KPIs. "
            "Return STRICT JSON ONLY with exactly these keys: "
            "analysis (string), evidence (list of strings), kpis (object). "
            "Focus ONLY on the query-relevant signals. "
            "Do NOT summarize the entire filing."
        ),
        agent=analysis_agent,
        expected_output="analysis_json",
        output_key="analysis_json",
        inputs={
            "retrieved_chunks": "{{ retrieved_chunks }}",
            "query": query,
            "kpis": "{{ kpi_json.kpis }}",
            "evidence": "{{ kpi_json.evidence }}"
        }
    )
    
    # 3. Generation
    generation_task = Task(
        description=(
            "Your ONLY job is to write a concise Final Analyst Insight paragraph that DIRECTLY and ACCURATELY answers the user's exact query: '{query}'.\n\n"
            "CRITICAL INSTRUCTIONS - FOLLOW EXACTLY:\n"
            "1. If the query asks about risks, competitive position, challenges, vulnerabilities, threats, or competitive risks → you MUST use chunks from the 'Risk Factors' section. Explicitly mention at least three of the following: supply-chain constraints, regulatory uncertainty, market adoption risks, and technology development risks. Do NOT default to a financial or operational summary.\n"
            "2. If the query asks about innovation, R&D, trials, pipeline, or investment → prioritize 'Operational Highlights' and 'Key Operating Metrics' and mention clinical trials, NFX-7 expansion, Cognitive Systems Integration Program, and the 30.5% R&D investment ratio.\n"
            "3. If the query asks about financial performance → prioritize 'Financial Performance Summary' and include revenue, gross margin, operating income, cash, CCY, and adoption rate.\n"
            "4. Use ONLY the provided chunks. Do not invent facts or use general statements.\n"
            "5. Output exactly 5–7 complete sentences. End with a period. No titles, no bullet points, no extra text.\n"
            "6. If you cannot answer the query with the available chunks, say: 'Insufficient specific information found to fully answer this query.'\n"
        ),
        agent=generation_agent,
        expected_output="draft_insight",
        output_key="draft_insight",
        inputs={
            "analysis": "{{ analysis_json.analysis | tojson }}",
            "evidence": "{{ analysis_json.evidence | tojson }}",
            "query": query
        }
    )
            
    # 4. Critic
    critic_task = Task(
        description=(
        "Review the draft_insight and ensure it DIRECTLY answers the user's exact query: '{query}'.\n\n"
        "CRITICAL RULE - DO NOT IGNORE:\n"
        "If the query asks about risks, competitive challenges, vulnerabilities, or competitive position (even if it lists specific risks like supply-chain, regulatory, market adoption, or technology development) → you MUST use chunks from the 'Risk Factors' section and explicitly mention at least three of the following: supply-chain constraints, regulatory uncertainty, market adoption risks, and technology development risks. Do not default to a financial or operational summary. If the draft does not include risks, rewrite it to include them.\n"
        "Keep the output to 5–7 tight, factual sentences.\n"
        "Output ONLY the final polished insight as plain text."
    ),
        agent=critic_agent,
        expected_output="final_insight",
        output_key="final_insight",
        inputs={
            "draft_insight": "{{ draft_insight.raw | default('') }}",
            "analysis": "{{ analysis_json.analysis }}",
            "evidence": "{{ analysis_json.evidence }}",
            "query": query
        },
        return_output=True
    )
    # CREW
    crew = Crew(
        agents=[retrieval_agent, analysis_agent, generation_agent, critic_agent],
        tasks=[retrieval_task, kpi_extraction_task, analysis_task, generation_task, critic_task],
        process=Process.sequential,
        verbose=True,
        return_task_outputs=True
    )
    result = crew.kickoff()
    print("DEBUG ORCHESTRATOR RESULT >>>", result)
    print("\n--- KPI EXTRACTION RAW ---")
    print(result.tasks_output[1].raw)

    print("\n--- ANALYSIS RAW ---")
    print(result.tasks_output[2].raw)

    print("\n--- FINAL INSIGHT RAW ---")
    print(result.tasks_output[-1].raw)
    _ = result.tasks_output[:]  # force materialization

    final_output = result.tasks_output[-1].raw
    

    # -----------------------------
    # Extract KPI JSON (task index 1)
    # -----------------------------
    kpi_raw = result.tasks_output[1].raw

    # Strip markdown fences if present
    if isinstance(kpi_raw, str):
        cleaned = kpi_raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")          # remove backticks
            cleaned = cleaned.replace("json", "", 1).strip()
        kpi_raw = cleaned

    try:
        kpi_json = json.loads(kpi_raw) if isinstance(kpi_raw, str) else kpi_raw
    except:
        kpi_json = {"kpis": {}, "evidence": {}}

    # -----------------------------
    # Extract analysis JSON (task index 2)
    # -----------------------------
    analysis_raw = result.tasks_output[2].raw

    # Strip markdown fences if present
    if isinstance(analysis_raw, str):
        cleaned = analysis_raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")          # remove backticks
            cleaned = cleaned.replace("json", "", 1).strip()
        analysis_raw = cleaned

    try:
        analysis_json = json.loads(analysis_raw) if isinstance(analysis_raw, str) else analysis_raw
    except:
        analysis_json = {"analysis": analysis_raw, "kpis": {}, "evidence": []}

    # -----------------------------
    # Determine canonical metric
    # -----------------------------
    canonical_metric = None
    kpi_keys = list(analysis_json.get("kpis", {}).keys())

    # 1. If the query mentions a KPI name, use that
    query_lower = query.lower()
    for key in kpi_keys:
        if key.lower() in query_lower:
            canonical_metric = key
        break

    # 2. If still none and only one KPI exists, use it
    if canonical_metric is None and len(kpi_keys) == 1:
        canonical_metric = kpi_keys[0]

    # 3. If still none, fallback to the first KPI
    if canonical_metric is None and len(kpi_keys) > 0:
        canonical_metric = kpi_keys[0]

    # === STRICT RISK FORCING (only for clear risk queries) ===
    # Ensure final_insight always exists
    if "final_insight" not in locals() or final_insight is None:
        final_insight = ""

    query_lower = query.lower()
    is_risk_query = any(
        word in query_lower
        for word in ["risk", "challenge", "competitive", "vulnerab", "adverse", "position", "threat", "uncertainty"]
    )

    if is_risk_query:
        print(f"DEBUG: Risk query detected → '{query}'")

        if "risk" not in final_insight.lower() and "supply-chain" not in final_insight.lower():
            final_insight += (
                "\n\nKey risks facing the company include supply-chain constraints related to rare-earth materials, "
                "regulatory uncertainty regarding neural data privacy and cognitive-processing technologies, "
                "market adoption risks due to competitive pressures and macroeconomic conditions, "
                "and technology development risks that could impact future competitive positioning."
            )
            print("DEBUG: Risk paragraph was APPENDED")
        else:
            print("DEBUG: Risk paragraph already present — skipped")

    else:
        print(f"DEBUG: Non-risk query → '{query}' (no risk paragraph added)")

    # -----------------------------
    # Return final structured output
    # -----------------------------
    return {
        "final_output": final_output,
        "canonical_metric": canonical_metric,
        "kpis": analysis_json.get("kpis", {}),
        "kpi_evidence": analysis_json.get("evidence", []),
        "raw_kpis": kpi_json.get("kpis", {}),
        "raw_kpi_evidence": kpi_json.get("evidence", {})
    }

# ---------------------------------------------------------
# PUBLIC ENTRYPOINT
# ---------------------------------------------------------

def run_financial_analysis(company, query: str):
    result = run_single_company_flow(company, query)
    
    # Extract ONLY the final summary text
    final_summary = result.get("final_output", "")

    # Return a clean, minimal structure
    return {
        "final_output": final_summary,
        #"canonical_metric": result.get("canonical_metric"),
        "kpis": result.get("kpis", {}),
        "kpi_evidence": result.get("kpi_evidence", []),
        "tasks_output": result.get("tasks_output", [])
    }