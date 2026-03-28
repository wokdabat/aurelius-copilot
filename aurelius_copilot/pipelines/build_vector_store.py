# pipelines/build_vector_store.py
import os
import shutil
from typing import List, Dict
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import pickle

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FILINGS_DIR = os.path.join(DATA_DIR, "filings")
VECTOR_STORE_PATH = os.path.join(DATA_DIR, "vector_store")
BM25_INDEX_PATH = os.path.join(DATA_DIR, "bm25_index.pkl")

SECTION_HEADERS = [
    "Business Overview",
    "Key Operating Metrics",
    "Financial Performance Summary",
    "Operational Highlights",
    "Risk Factors",
    "Outlook and Forward Looking Statements"
]

embeddings = SentenceTransformer("all-MiniLM-L6-v2")


# ------------------------------------------------------------
# CLEAN LINE
# ------------------------------------------------------------

def clean_line(raw: str) -> str:
    if raw is None:
        return ""
    return raw.strip()


# ------------------------------------------------------------
# PARSE MARKDOWN
# ------------------------------------------------------------

def parse_markdown(filepath: str, company: str, year: str, filing_type: str) -> List[Dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    chunks = []
    current_section = None
    current_text = []
    chunk_counter = 0

    in_header_block = True

    for line in lines:
        stripped = line.strip()

        if not stripped:
            if current_text and not in_header_block:
                text = "\n".join(current_text).strip()
                if len(text) > 60:  # allow smaller chunks for bullets
                    chunks.append({
                        "id": f"{company}_{year}_{filing_type}_{chunk_counter}",
                        "text": text,
                        "metadata": {
                            "company": company,
                            "year": year,
                            "filing_type": filing_type,
                            "section": current_section or "Uncategorized"
                        }
                    })
                    chunk_counter += 1
                current_text = []
            continue

        # Skip initial header/metadata block
        if in_header_block:
            if stripped in SECTION_HEADERS or stripped.startswith("Business Overview"):
                in_header_block = False
                current_section = stripped
            elif stripped.startswith(("Company:", "Filing Type:", "Fiscal Year:", "Regulatory Body:")):
                continue
            else:
                in_header_block = False
            continue

        # Detect section headers
        if stripped in SECTION_HEADERS:
            if current_text:
                text = "\n".join(current_text).strip()
                if len(text) > 60:
                    chunks.append({
                        "id": f"{company}_{year}_{filing_type}_{chunk_counter}",
                        "text": text,
                        "metadata": {
                            "company": company,
                            "year": year,
                            "filing_type": filing_type,
                            "section": current_section or "Uncategorized"
                        }
                    })
                    chunk_counter += 1
                current_text = []
            current_section = stripped
            continue

        # Split on bullet points — this is the key for Risk Factors and Operational Highlights
        if stripped.startswith(('- ', '**', '* ')):
            if current_text:
                text = "\n".join(current_text).strip()
                if len(text) > 60:
                    chunks.append({
                        "id": f"{company}_{year}_{filing_type}_{chunk_counter}",
                        "text": text,
                        "metadata": {
                            "company": company,
                            "year": year,
                            "filing_type": filing_type,
                            "section": current_section or "Uncategorized"
                        }
                    })
                    chunk_counter += 1
                current_text = []
            current_text.append(line)
            continue

        current_text.append(line)

        # Split long paragraphs
        if len(" ".join(current_text)) > 400:
            text = "\n".join(current_text).strip()
            chunks.append({
                "id": f"{company}_{year}_{filing_type}_{chunk_counter}",
                "text": text,
                "metadata": {
                    "company": company,
                    "year": year,
                    "filing_type": filing_type,
                    "section": current_section or "Uncategorized"
                }
            })
            chunk_counter += 1
            current_text = [line]

    # Save final chunk
    if current_text:
        text = "\n".join(current_text).strip()
        if len(text) > 60:
            chunks.append({
                "id": f"{company}_{year}_{filing_type}_{chunk_counter}",
                "text": text,
                "metadata": {
                    "company": company,
                    "year": year,
                    "filing_type": filing_type,
                    "section": current_section or "Uncategorized"
                }
            })

    print(f"Parsed {len(chunks)} granular chunks from {filepath}")
    return chunks

# ------------------------------------------------------------
# BUILD CHROMA VECTOR STORE
# ------------------------------------------------------------

def build_chroma(all_chunks):
    print("📦 Building Chroma vector store...")

    # Remove old store
    if os.path.exists(VECTOR_STORE_PATH):
        print("🗑️ Removing old vector store...")
        shutil.rmtree(VECTOR_STORE_PATH)

    # Create Chroma client
    client = PersistentClient(path=VECTOR_STORE_PATH)

    # Embedding function (supported in 0.4.24)
    from chromadb.utils import embedding_functions
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Create collection WITH embedding function
    collection = client.get_or_create_collection(
        name="filings",
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"}
    )

    # Add chunks
    collection.add(
        ids=[c["id"] for c in all_chunks],
        documents=[c["text"] for c in all_chunks],
        metadatas=[c["metadata"] for c in all_chunks]
    )

    print("✅ Chroma vector store built.")

# ------------------------------------------------------------
# BUILD BM25
# ------------------------------------------------------------

def build_bm25(chunks: List[Dict]):
    print("📦 Building BM25 index...")

    from rank_bm25 import BM25Okapi

    # Tokenize using the text field
    corpus = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(corpus)

    # Store full structured chunks (text + metadata)
    bm25_data = {
        "bm25": bm25,
        "chunks": chunks,   # <-- structured, not just strings
    }

    print("📝 Writing BM25 to:", BM25_INDEX_PATH)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25_data, f)

    print("✅ BM25 index saved.")

# ------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------

def main():
    print("🚀 Starting ingestion pipeline...")

    all_chunks = []

    for filename in os.listdir(FILINGS_DIR):
        if not filename.endswith(".md"):
            continue

        print(f"📄 Processing {filename}...")

        parts = filename.replace(".md", "").split("_")
        company = parts[0]
        year = parts[1]
        filing_type = parts[2]

        filepath = os.path.join(FILINGS_DIR, filename)
        chunks = parse_markdown(filepath, company, year, filing_type)
        all_chunks.extend(chunks)

    print(f"✅ Total chunks loaded: {len(all_chunks)}")

    # Assign deterministic unique IDs to each chunk
    for idx, chunk in enumerate(all_chunks):
        chunk["id"] = f"chunk_{idx}"

    build_chroma(all_chunks)
    build_bm25(all_chunks)

    print("🎉 Ingestion complete.")


if __name__ == "__main__":
    main()