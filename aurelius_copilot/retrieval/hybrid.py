# retrieval/hybrid.py
import os
import pickle
import re
import numpy as np
from typing import List, Dict, Any
from aurelius_copilot.pipelines.build_vector_store import BM25_INDEX_PATH
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "data", "vector_store")
BM25_PATH = BM25_INDEX_PATH

# ---------------------------------------------------------
# LOAD VECTOR STORE (CHROMA)
# ---------------------------------------------------------

client = PersistentClient(path=VECTOR_STORE_PATH)
collection = client.get_collection("filings")

# ---------------------------------------------------------
# LOAD BM25 + METADATA
# ---------------------------------------------------------
print("🔍 Loading BM25 from:", BM25_PATH)
with open(BM25_PATH, "rb") as f:
    bm25_data = pickle.load(f)

bm25 = bm25_data["bm25"]
bm25_chunks = bm25_data["chunks"]

# ---------------------------------------------------------
# LOAD EMBEDDING MODEL
# ---------------------------------------------------------

embeddings = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------------------------------------
# FINANCIAL RETRIEVAL BOOSTING CONFIG
# ---------------------------------------------------------

FINANCIAL_SECTIONS = [
    "financial performance summary",
    "key operating metrics",
    "financial statements",
    "results of operations",
    "management discussion",
    "md&a",
    "selected financial data",
    "consolidated statements",
    "statement of operations",
    "income statement",
]

FINANCIAL_TERMS = [
    "total revenue",
    "revenue",
    "gross margin",
    "operating income",
    "cash and equivalents",
    "r&d investment ratio",
    "neural interface adoption rate",
    "production efficiency index",
    "ccy",
    "cognitive compute yield",
    "compute yield",
    "neural-processing efficiency",
    "rare-earth material optimization",
]

# ---------------------------------------------------------
# SCORING FUNCTION
# ---------------------------------------------------------

def score_chunk(query, chunk_text, metadata, embed_score, bm25_score):
    """Compute a financial-aware hybrid score."""

    # Rebalanced weights: BM25 > semantic
    score = 0.15 * embed_score + 0.55 * bm25_score

    ct = chunk_text.lower()
    section_title = metadata.get("section", "") or metadata.get("section_title", "")
    st = section_title.lower() if section_title else ""

    # Strong numeric boost
    if re.search(r"\d", chunk_text):
        score += 6.0

    # Section-specific boosts
    if "financial performance summary" in st:
        score += 12.0

    if "key operating metrics" in st:
        score += 8.0

    # General financial section boost
    if any(fs in st for fs in FINANCIAL_SECTIONS):
        score += 6.0

    # Financial term boost
    if any(term in ct for term in FINANCIAL_TERMS):
        score += 6.0

    # CCY-specific targeted boost
    if "ccy" in ct or "cognitive compute yield" in ct:
        score += 4.0

    # Penalize business overview
    if "business overview" in st:
        score -= 6.0

    # Penalize outlook
    if "outlook" in st or "forward looking" in st:
        score -= 6.0

    return score

# ---------------------------------------------------------
# HYBRID RETRIEVAL
# ---------------------------------------------------------
from sentence_transformers import CrossEncoder

def hybrid_retrieve(query: str, company: str, top_k: int = 10) -> List[Dict[str, Any]]:
    print("\n" + "="*100)
    print(f"🔍 HYBRID RETRIEVAL - Query: '{query}'")
    print(f"Company filter: '{company}'")
    print("="*100)

    def matches_company(meta):
        if not company:
            return True
        if "company" in meta and meta["company"].lower() == company.lower():
            return True
        if "source" in meta and meta["source"].lower().startswith(company.lower()):
            return True
        return False

    # 1. Semantic search
    query_vec = embeddings.encode(query).tolist()
    semantic_results = collection.query(
        query_embeddings=[query_vec],
        n_results=200,  # more candidates
        where={"company": company.lower()}
    )

    semantic_chunks = []  # always defined

    if "documents" in semantic_results and semantic_results["documents"] and len(semantic_results["documents"][0]) > 0:
        for i in range(len(semantic_results["documents"][0])):
            meta = semantic_results["metadatas"][0][i]
            if matches_company(meta):  # fallback safety
                semantic_chunks.append({
                    "content": semantic_results["documents"][0][i],
                    "metadata": meta,
                    "semantic_score": float(semantic_results["distances"][0][i]) if "distances" in semantic_results else 999.0
                })

    print(f"Semantic chunks after filter: {len(semantic_chunks)}")

    # Rerank semantic chunks if any
    if semantic_chunks:
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
        pairs = [[query, c["content"]] for c in semantic_chunks]
        rerank_scores = reranker.predict(pairs)
        for c, score in zip(semantic_chunks, rerank_scores):
            c["rerank_score"] = float(score)

        semantic_chunks = sorted(semantic_chunks, key=lambda x: x.get("rerank_score", 0), reverse=True)

    # 2. BM25 (simplified, company-filtered first)
    company_filtered_chunks = [entry for entry in bm25_chunks if matches_company(entry["metadata"])]
    print(f"BM25 chunks after filter: {len(company_filtered_chunks)}")

    bm25_results = []
    if company_filtered_chunks:
        tokens = set(word.lower() for word in query.split() if len(word) > 2)
        bm25_scores = bm25.get_scores(tokens)
        for i, entry in enumerate(bm25_chunks):
            if entry in company_filtered_chunks:
                bm25_results.append({
                    "content": entry["text"],
                    "metadata": entry["metadata"],
                    "bm25_score": float(bm25_scores[i])
                })

    # 3. Fusion with penalties
    fused = {}

    for c in semantic_chunks:
        key = c["metadata"].get("id") or c["content"][:80]
        fused[key] = {
            "content": c["content"],
            "metadata": c["metadata"],
            "semantic": c["semantic_score"],
            "bm25": 0.0,
            "rerank_score": c.get("rerank_score", 0.0)
        }

    for c in bm25_results:
        key = c["metadata"].get("id") or c["content"][:80]
        if key not in fused:
            fused[key] = {
                "content": c["content"],
                "metadata": c["metadata"],
                "semantic": 999.0,
                "bm25": c["bm25_score"],
                "rerank_score": 0.0
            }
        else:
            fused[key]["bm25"] = c["bm25_score"]

    # Compute hybrid_score + keyword boost for every item
    for item in fused.values():
        semantic_sim = 1.0 - item["semantic"] if item["semantic"] < 999 else 0.0
        #rerank_bonus = item["rerank_score"] / 10 if item["rerank_score"] else 0.0
        bm25_contrib = item["bm25"] if item["bm25"] > 0 else 0.0

        penalty = 0.0
        if item["metadata"].get("section") == "Uncategorized":
            penalty = 0.6
        if item["metadata"].get("section") == "Business Overview":
            penalty += 12.0  # very heavy penalty to push overview down
        if len(item["content"]) < 150:
            penalty += 0.3

        # Base rerank_score (from cross-encoder)
        base_rerank = item.get("rerank_score", 0.0)

        # Very aggressive boost for Risk Factors when query is about competitive position or risks
        section = item["metadata"].get("section", "")
        if item["metadata"].get("section") == "Business Overview":
            continue  # skip this chunk entirely for retrieval
        if item["metadata"].get("section") == "Risk Factors":
            if any(word in query.lower() for word in ["risk", "challenge", "competitive", "adverse", "vulnerab", "position", "threat", "uncertainty"]):
                item["rerank_score"] += 20.0  # extremely aggressive
            else:
                item["rerank_score"] += 8.0   

        # Keep the other section boosts if you have them
        elif section in ["Operational Highlights", "Key Operating Metrics"]:
            if any(word in query.lower() for word in ["r&d", "innovation", "trial", "investment"]):
                item["rerank_score"] += 8.0

        elif section == "Financial Performance Summary":
            if any(word in query.lower() for word in ["revenue", "financial", "margin", "income", "cash"]):
                item["rerank_score"] += 8.0

        # Final rerank_score (do NOT overwrite with rerank_bonus)
        item["rerank_score"] = base_rerank

        # Optional: still compute hybrid_score for logging/debug
        semantic_sim = 1.0 - item["semantic"] if item["semantic"] < 999 else 0.0
        bm25_contrib = item["bm25"] if item["bm25"] > 0 else 0.0
        item["hybrid_score"] = (0.8 * semantic_sim + 0.1 * bm25_contrib) * (1 - penalty)

        # Inside the for item in fused.values() loop, after setting hybrid_score
        query_words = set(word.lower() for word in query.split() if len(word) > 3)
        content_lower = item["content"].lower()
        keyword_hits = sum(1 for w in query_words if w in content_lower)
        keyword_bonus = 3.0 * keyword_hits  # aggressive
        item["hybrid_score"] += keyword_bonus
        item["hybrid_score"] = min(item["hybrid_score"], 5.0)  # cap

    # Sort purely by rerank_score (query relevance now completely overrides everything else)
    ranked = sorted(fused.values(), key=lambda x: x.get("rerank_score", -999), reverse=True)[:50]

    print(f"Final ranked chunks: {len(ranked)}")
    if ranked:
        print(f"Top rerank_score: {ranked[0].get('rerank_score', 'N/A')}")
        print(f"Top chunk: {ranked[0]['content'][:200]}...")
    else:
        print("No ranked chunks — returning empty")

    print("="*100 + "\n")
    return ranked

def debug_hybrid_retrieve(query: str, company: str, top_k: int = 10):
    """Debug version of hybrid_retrieve that prints scoring details."""

    print("\n" + "="*120)
    print(f"🔍 DEBUG HYBRID RETRIEVAL")
    print(f"Query: {query}")
    print(f"Company: {company}")
    print("="*120)

    results = hybrid_retrieve(query, company, top_k=50)

    for i, item in enumerate(results[:top_k]):
        meta = item["metadata"]
        section = meta.get("section") or meta.get("section_title") or "UNKNOWN"
        content_preview = item["content"][:200].replace("\n", " ")

        print(f"\n[{i+1}] Score: {item['hybrid_score']:.4f}")
        print(f"   Section: {section}")
        print(f"   Semantic: {item.get('semantic', 0):.4f}")
        print(f"   BM25: {item.get('bm25', 0):.4f}")
        print(f"   Content: {content_preview}")

    print("\n" + "="*120)
    print("END DEBUG OUTPUT")
    print("="*120 + "\n")

    return results[:top_k]