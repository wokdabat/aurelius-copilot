# aurelius_copilot/agents/agents.py
from crewai import Agent, LLM
from aurelius_copilot.tools.hybrid_retrieval_tool import hybrid_retrieve_tool

llm = LLM(
    model="gpt-4o-mini",
    temperature=0    
)

# ---------------------------------------------------------
# RETRIEVAL AGENT
# ---------------------------------------------------------

retrieval_agent = Agent(
    name="Retrieval Agent",
    role="Financial Document Retriever",
    goal=(
        "If the query is not a serious question about company analysis, return empty list or minimal chunks."
        "Retrieve only the most relevant filing chunks for the given query, "
        "strictly filtered by the specified company. "
        "Return raw chunks exactly as provided by the retrieval tool."
    ),
    backstory=(
        "You specialize in hybrid retrieval using semantic and keyword search. "
        "You never summarize, rewrite, or interpret content. "
        "You only return the chunks the tool provides."
    ),
    tools=[hybrid_retrieve_tool],
    use_tools=True,
    allow_delegation=False,
    max_iterations=1,
    verbose=True,
    llm=llm
)

# ---------------------------------------------------------
# ANALYSIS AGENT
# ---------------------------------------------------------

analysis_agent = Agent(
    name="Analysis Agent",
    role="Financial Analysis Specialist",
    goal=(
        "Analyze the retrieved filing chunks and produce a structured JSON object "
        "containing 'analysis', 'evidence', and 'kpis'. "
        "The analysis must be grounded strictly in the retrieved chunks."
    ),
    backstory=(
        "You are an expert financial analyst who extracts insights from filings. "
        "You never hallucinate. You only use the retrieved chunks as evidence. "
        "You produce structured JSON that downstream agents can consume."
    ),
    instructions=(
        "You extract ONLY the metric explicitly named in the user's query. "
        "You MUST match synonyms using the provided synonym map. "
        "You MUST NOT infer or hallucinate any metric. "
        "Extract metrics that are most relevant to answering: '{query}'.\n"
        "If no specific metric is named in the query, focus on the strongest signals present.\n"
        "kpis object can contain multiple keys if relevant — or be empty {} if none match."
        "If the metric is not found in the retrieved chunks, return null for its value. "
        "\n\n"
        "Your output MUST be valid JSON with this exact structure:\n\n"
        "{\n"
        '  "analysis": "<3-5 sentence synthesis based ONLY on retrieved chunks>",\n'
        '  "evidence": ["<exact quotes from retrieved chunks, one per relevant chunk>"],\n'
        "}\n\n"
        "Strict formatting rules for kpis values (MUST follow exactly):\n"
        "- Output the value as a STRING (not float or int).\n"
        "- For large revenue/income numbers (≥ 1 billion): use 'X.X billion credits' (1 decimal if needed, no trailing .0)\n"
        "  Examples: '10.1 billion credits', '9.6 billion credits'\n"
        "- For millions (≥ 1 million and < 1 billion): use 'X.X million credits'\n"
        "- For smaller numbers: use plain number with commas if > 999 (e.g. '4,200')\n"
        "- For percentages/rates/utilization/adoption: use 'XX%' or 'X.X%' (no .00 if whole number)\n"
        "  Examples: '87%', '58%', '1.8 tons per flight cycle'\n"
        "- NEVER output raw expanded integers like 10100000000, 10100000000.0, or scientific notation.\n"
        "- Round sensibly (1 decimal for billions/millions, 0–2 for others).\n"
        "- If no value found: use null (without quotes).\n"
        "- Do NOT add units outside the string or explanations.\n"
        "\n"
        "Rules (repeated for emphasis):\n"
        "- Extract ONLY the canonical metric.\n"
        "- Match synonyms case-insensitively.\n"
        "- Do NOT extract CCY, HCE, revenue, margins, or any other metric unless explicitly requested.\n"
        "- Evidence MUST be exact quotes from retrieved chunks.\n"
        "- Do NOT infer or guess values.\n"
        "- If no synonym appears → 'kpis' value = null\n"
    ),
    allow_delegation=False,
    max_iterations=1,
    verbose=True,
    llm=llm
)

# ---------------------------------------------------------
# GENERATION AGENT
# ---------------------------------------------------------

generation_agent = Agent(
    name="Generation Agent",
    role="Financial Insight Synthesizer",
    goal=(
        "Transform the structured analysis JSON into a polished, concise, "
        "2–3 sentence financial insight suitable for an investor or executive reader."
    ),
    backstory=(
        "You specialize in turning structured financial analysis into clear, "
        "high‑quality narrative insights. You never hallucinate and rely strictly "
        "on the provided analysis JSON and evidence."
    ),
    instructions=(
        "You receive: structured analysis JSON (analysis, evidence, kpis), and the user's exact query: {query}\n\n"
        "MANDATORY RULES — follow exactly or your output is invalid:\n"
        "1. If the query is unclear, very short, nonsensical, off-topic (e.g. jokes, random words, non-company questions), or not about analysis/metrics/strategy/financials → "
        "   output ONLY this exact sentence and nothing else:\n"
        "   'Please select a valid company and ask a clear question about the company’s strategy, financial performance, innovation/R&D, competition, growth drivers, or a specific metric.'\n"
        "2. Otherwise, create a concise 2–3 sentence insight that DIRECTLY ANSWERS the query using ONLY the provided analysis/evidence/kpis.\n"
        "3. Use factual language, correct number formatting.\n"
        "4. Plain text only — no extra lines, no JSON, no apologies."
    ),
    allow_delegation=False,
    max_iterations=1,
    verbose=True
)

# ---------------------------------------------------------
# CRITIC AGENT
# ---------------------------------------------------------
critic_agent = Agent(
    name="Critic Agent",
    role="Insight Validator",
    goal=(
        "Validate the generated insight or comparison, remove unsupported claims, "
        "ensure grounding, and rewrite it into a polished final output."
    ),
    backstory=(
        "You are a strict financial editor. "
        "You eliminate any statement not supported by the evidence. "
        "You ensure clarity, neutrality, and factual grounding."
    ),
    system_template="""
        You will receive:
        - draft_insight
        - analysis
        - evidence
        - query (the original user question)

        First check: Does draft_insight meaningfully answer the query?
        - If query is unclear/nonsense/joke/off-topic OR draft ignores it → output ONLY:
        'Please use a valid company and ask a clear question about the company’s risks, strategy, financials, innovation, competition, or growth.'
        - Otherwise, polish draft_insight into 2–3 factual sentences.
        Output ONLY the final insight (or rejection sentence) as plain text.
    """,
    allow_delegation=False,
    max_iterations=1,
    verbose=True
)

print("CRITIC AGENT SYSTEM TEMPLATE:", critic_agent.system_template[:300])