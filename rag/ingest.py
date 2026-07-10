from sentence_transformers import SentenceTransformer
import chromadb
import os

# ── Load JD corpus ─────────────────────────────────────────────────────────────
corpus_path = os.path.join(os.path.dirname(__file__), "jd_corpus.txt")
with open(corpus_path, "r") as f:
    raw_text = f.read()

# ── Chunk by double newline, keep chunks > 80 chars ───────────────────────────
chunks = [c.strip() for c in raw_text.split("\n\n") if len(c.strip()) > 80]
print(f"Total chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"  [{i}] {chunk[:80]}...")

# ── Embed ──────────────────────────────────────────────────────────────────────
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks).tolist()
print(f"\nEmbeddings generated: {len(embeddings)} vectors of dim {len(embeddings[0])}")

# ── Store in persistent ChromaDB ───────────────────────────────────────────────
db_path = os.path.join(os.path.dirname(__file__), "chroma_db")
client = chromadb.PersistentClient(path=db_path)

# Fresh start — delete if exists
try:
    client.delete_collection("jd_knowledge")
    print("Cleared existing collection")
except:
    pass

collection = client.get_or_create_collection("jd_knowledge")
collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

print(f"\nStored {collection.count()} chunks in ChromaDB at {db_path} ✓")