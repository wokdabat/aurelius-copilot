# pipelines/build_bm25_index.py

import os
import pickle
from rank_bm25 import BM25Okapi

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILINGS_DIR = os.path.join(BASE_DIR, "data", "filings")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "bm25_index.pkl")

def load_all_chunks():
    chunks = []
    for filename in os.listdir(FILINGS_DIR):
        if filename.endswith(".md"):
            path = os.path.join(FILINGS_DIR, filename)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

            # Extract company
            company = filename.split("_")[0].title()

            for i, p in enumerate(paragraphs):
                chunks.append({
                    "text": p,
                    "metadata": {
                        "id": f"{filename}_{i}",
                        "source": filename,
                        "company": company,
                        "chunk_index": i
                    }
                })
    return chunks

def build_bm25_index():
    print("Loading filings...")
    chunks = load_all_chunks()
    print(f"Loaded {len(chunks)} chunks.")

    tokenized = [c["text"].lower().split() for c in chunks]

    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized)

    data = {
        "bm25": bm25,
        "chunks": chunks
    }

    print(f"Saving BM25 index to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(data, f)

    print("BM25 index built successfully!")

if __name__ == "__main__":
    build_bm25_index()