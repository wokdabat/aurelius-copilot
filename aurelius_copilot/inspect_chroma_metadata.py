#inspect_chroma_metadata.py

from chromadb import PersistentClient
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "data", "vector_store")

client = PersistentClient(path=VECTOR_STORE_PATH)
collection = client.get_collection("filings")

results = collection.get(include=["metadatas", "documents"], limit=9999)

print(f"Total docs: {len(results['metadatas'])}")

for meta, doc in zip(results["metadatas"], results["documents"]):
    company = meta.get("company")
    if company is None:
        print("\n❌ Found a chunk with missing company metadata:")
        print("ID:", meta.get("id"))
        print("Source:", meta.get("source"))
        print("Chunk index:", meta.get("chunk_index"))
        print("Preview:", doc[:120])