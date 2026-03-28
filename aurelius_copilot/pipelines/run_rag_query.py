# pipelines/run_rag_query.py
import chromadb
from sentence_transformers import SentenceTransformer

VECTOR_DB_PATH = "data/vector_store"

# Load embedding model
model = SentenceTransformer("all-mpnet-base-v2")

# Connect to Chroma
client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

# Load or create the collection
collection = client.get_or_create_collection(
    name="filings",
    metadata={"hnsw:space": "cosine"}
)

from rank_bm25 import BM25Okapi
import pickle

# Load BM25 index
with open("data/bm25_index.pkl", "rb") as f:
    bm25_data = pickle.load(f)

bm25 = bm25_data["bm25"]
bm25_chunks = bm25_data["chunks"]

def hybrid_retrieve(query, company=None, n=8, alpha=0.5):
    """
    alpha = weight for semantic search
    (1 - alpha) = weight for BM25 keyword search
    """

    # 1. Semantic search (Chroma)
    query_embedding = model.encode(query).tolist()
    where = {"company": company} if company else None

    semantic_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n * 2,
        where=where
    )

    semantic_docs = semantic_results["documents"][0]
    semantic_metas = semantic_results["metadatas"][0]
    semantic_scores = [1 - d for d in semantic_results["distances"][0]]

    # 2. Keyword search (BM25)
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    bm25_ranked = sorted(
        zip(bm25_scores, bm25_chunks),
        key=lambda x: x[0],
        reverse=True
    )[:n * 2]

    # 3. Fusion
    fused = {}

    for doc, meta, score in zip(semantic_docs, semantic_metas, semantic_scores):
        key = meta["id"]
        fused[key] = fused.get(key, 0) + alpha * score

    for score, chunk in bm25_ranked:
        key = chunk["metadata"]["id"]
        fused[key] = fused.get(key, 0) + (1 - alpha) * score

    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:n]

    # 4. Return final chunks
    final_chunks = []
    for key, _ in ranked:
        for doc, meta in zip(semantic_docs, semantic_metas):
            if meta["id"] == key:
                final_chunks.append({"text": doc, "metadata": meta})
                break
        else:
            for _, chunk in bm25_ranked:
                if chunk["metadata"]["id"] == key:
                    final_chunks.append(chunk)
                    break

    return final_chunks

def retrieve_chunks(query, company=None, n=8):
    print("COLLECTION COUNT:", collection.count())
    print("SAMPLE METADATA:", collection.get(limit=1))
    print("DEBUG retrieve_chunks CALLED WITH:", query, company)

    # If company is specified, bypass embeddings entirely
    if company:
        results = collection.get(
            where={"company": company},
            include=["documents", "metadatas"],
            limit=n
        )

        docs = results.get("documents", [])
        metas = results.get("metadatas", [])

        return [{"text": d, "metadata": m} for d, m in zip(docs, metas)]

    # Otherwise fall back to embedding search
    query_embedding = model.encode(query).tolist()
    print("QUERY EMBEDDING NORM:", (sum(x*x for x in query_embedding) ** 0.5))

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
        include=["documents", "metadatas"]
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    return [{"text": d, "metadata": m} for d, m in zip(docs, metas)]