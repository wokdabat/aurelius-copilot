#aurelius_copilot/hybrid_retrieval_tool.py
from crewai.tools import tool
from aurelius_copilot.retrieval.hybrid import hybrid_retrieve

@tool
def hybrid_retrieve_tool(query: str, company: str):
    """Retrieve relevant filing chunks using financial-aware hybrid retrieval."""

    print("\n" + "="*80)
    print("🔎 HYBRID RETRIEVAL TOOL — START")
    print(f"Company: {company}")
    print(f"Query: {query}")
    print("="*80)

    chunks = hybrid_retrieve(query=query, company=company)

    if not chunks:
        print(f"❌ Retrieval returned no chunks for company={company}")
    else:
        print(f"🔎 Retrieved {len(chunks)} chunks for {company}")
        print(f"🔎 First chunk preview:\n{str(chunks[0])[:400]}")

    print("="*80 + "\n")

    # 🔥 FIX: CrewAI requires dict output
    return {"results": chunks}