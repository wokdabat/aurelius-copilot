# test_retrieval.py

import os
from pathlib import Path
from aurelius_copilot.retrieval.hybrid import hybrid_retrieve
from chromadb import PersistentClient
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_STORE_PATH = Path(__file__).resolve().parents[1] / "data" / "vector_store"
BM25_PATH = Path(__file__).resolve().parents[1] / "data" / "bm25_index.pkl"

def test_chroma():
    print("\n--- Testing Chroma Vector Store ---")
    client = PersistentClient(path=VECTOR_STORE_PATH)
    collection = client.get_collection("filings")

    results = collection.query(
        query_texts=["revenue growth"],
        n_results=5
    )

    print(f"Chroma returned {len(results['documents'][0])} results.")
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        print(f"- {meta.get('source')} | {doc[:60]}...")

def test_bm25():
    print("\n--- Testing BM25 Index ---")
    with open(BM25_PATH, "rb") as f:
        data = pickle.load(f)

    bm25 = data["bm25"]
    chunks = data["chunks"]

    tokens = "revenue growth".lower().split()
    scores = bm25.get_scores(tokens)

    top_idx = scores.argsort()[-5:][::-1]

    print(f"BM25 returned top 5 chunks:")
    for idx in top_idx:
        meta = chunks[idx]["metadata"]
        text = chunks[idx]["text"]
        print(f"- {meta.get('source')} | {text[:60]}...")

def test_hybrid():
    print("\n--- Testing Hybrid Retrieval ---")
    results = hybrid_retrieve("revenue growth", "NeuroFlux")

    print(f"Hybrid returned {len(results)} results.")
    for r in results:
        meta = r["metadata"]
        print(f"- {meta.get('company')} | {meta.get('source')} | score={r['hybrid_score']:.4f}")
        print(f"  {r['content'][:80]}...\n")

if __name__ == "__main__":
    test_chroma()
    test_bm25()
    test_hybrid()