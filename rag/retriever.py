# rag/retriever.py
# Query the ChromaDB and return relevant JD chunks

from sentence_transformers import SentenceTransformer
import chromadb
import os

model = SentenceTransformer("all-MiniLM-L6-v2")

db_path = os.path.join(os.path.dirname(__file__), "chroma_db")
client = chromadb.PersistentClient(path=db_path)
collection = client.get_collection("jd_knowledge")

def retrieve_context(query: str, k: int = 3) -> list:
    query_emb = model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_emb], n_results=k)
    return results["documents"][0]

# Test it
if __name__ == "__main__":
    queries = [
        "system design and distributed systems",
        "product metrics and roadmap",
        "machine learning and model training",
    ]

    for q in queries:
        print(f"\nQuery: '{q}'")
        chunks = retrieve_context(q, k=2)
        for i, chunk in enumerate(chunks):
            print(f"  [{i+1}] {chunk[:100]}...")