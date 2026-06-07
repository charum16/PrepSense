# practice/embeddings_rag.py
# Day 3 — Embeddings & RAG v1 Practice

from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np

# ── PART 1: Generate & explore embeddings ─────────────────────────────────────

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    # tech cluster
    "Python is a programming language",
    "JavaScript is used for web development",
    "Machine learning models need training data",
    "Neural networks have multiple layers",
    "Deep learning is a subset of machine learning",
    # food cluster
    "Pizza is a popular Italian dish",
    "Sushi is a traditional Japanese food",
    "Pasta is made from wheat flour",
    "Coffee is a caffeinated beverage",
    "Tea is consumed worldwide",
    # sports cluster
    "Football is played with 11 players",
    "Cricket is popular in South Asia",
    "Basketball involves shooting into a hoop",
    "Swimming is an Olympic sport",
    "Tennis is played on various surfaces",
    # random
    "The moon orbits the Earth",
    "Climate change affects global temperatures",
    "Music therapy reduces anxiety",
    "Reading improves vocabulary",
    "Sleep is essential for memory consolidation",
]

embeddings = model.encode(sentences)
print(f"Shape: {embeddings.shape}")  # (20, 384)
print(f"First embedding (first 5 dims): {embeddings[0][:5]}")

# ── PART 2: Cosine similarity — see which sentences are "close" ───────────────

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

query = "I love coding in Python"
query_embedding = model.encode(query)

print(f"\nQuery: '{query}'")
print("Top 3 most similar sentences:")

scores = [(sentences[i], cosine_similarity(query_embedding, embeddings[i])) for i in range(len(sentences))]
scores.sort(key=lambda x: x[1], reverse=True)
for sent, score in scores[:3]:
    print(f"  {score:.3f} — {sent}")

# ── PART 3: Store in ChromaDB & do semantic search (RAG v1 for PrepSense) ─────

# 5 mini job description chunks (simulating your rag/jd_corpus.txt)
jd_chunks = [
    "Amazon SWE: Strong fundamentals in data structures, algorithms, and system design. Experience with distributed systems at scale.",
    "Amazon SWE: Demonstrates leadership principles especially Dive Deep and Deliver Results. Owns problems end to end.",
    "Google SWE: Expertise in large-scale system architecture, reliability, and performance optimization.",
    "Google SWE: Strong coding skills in Python, Java, or C++. Passion for clean, testable, and maintainable code.",
    "Meta PM: Define product vision and roadmap. Work cross-functionally with engineering, design, and data science.",
    "Meta PM: Analyze product metrics and user research to drive decisions. Strong communication with stakeholders.",
    "Data Engineer: Build and maintain scalable data pipelines. Proficiency in SQL, Spark, and cloud platforms (AWS/GCP).",
    "Data Engineer: Design data models and ETL processes. Experience with Airflow and dbt preferred.",
]

# Embed and store in ChromaDB
client = chromadb.Client()  # in-memory for practice (no PersistentClient needed)
collection = client.get_or_create_collection("practice_jds")

jd_embeddings = model.encode(jd_chunks).tolist()
collection.add(
    documents=jd_chunks,
    embeddings=jd_embeddings,
    ids=[f"jd_{i}" for i in range(len(jd_chunks))]
)
print(f"\nStored {len(jd_chunks)} JD chunks in ChromaDB ✓")

# Semantic search — this is what your question generator will do
def retrieve_context(query: str, k: int = 3) -> list:
    query_emb = model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_emb], n_results=k)
    return results["documents"][0]

# Test retrieval
test_queries = [
    "system design and distributed systems",
    "product metrics and roadmap",
    "data pipelines and SQL",
]

for q in test_queries:
    print(f"\nQuery: '{q}'")
    retrieved = retrieve_context(q)
    for i, chunk in enumerate(retrieved):
        print(f"  [{i+1}] {chunk[:80]}...")


# practice/embeddings_viz.py
# Day 3 — Visualizing embedding space with PCA

from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "Python is a programming language",
    "JavaScript is used for web development",
    "Machine learning models need training data",
    "Neural networks have multiple layers",
    "Deep learning is a subset of machine learning",
    "Pizza is a popular Italian dish",
    "Sushi is a traditional Japanese food",
    "Pasta is made from wheat flour",
    "Coffee is a caffeinated beverage",
    "Tea is consumed worldwide",
    "Football is played with 11 players",
    "Cricket is popular in South Asia",
    "Basketball involves shooting into a hoop",
    "Swimming is an Olympic sport",
    "Tennis is played on various surfaces",
    "The moon orbits the Earth",
    "Climate change affects global temperatures",
    "Music therapy reduces anxiety",
    "Reading improves vocabulary",
    "Sleep is essential for memory consolidation",
]

labels = ["tech"]*5 + ["food"]*5 + ["sports"]*5 + ["other"]*5
colors = {"tech": "blue", "food": "green", "sports": "red", "other": "gray"}

embeddings = model.encode(sentences)

# Reduce 384 dims → 2 dims for visualization
pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

# Plot
plt.figure(figsize=(12, 8))
for i, (x, y) in enumerate(reduced):
    color = colors[labels[i]]
    plt.scatter(x, y, color=color, s=100)
    plt.annotate(sentences[i][:30], (x, y), fontsize=7, alpha=0.8)

# Legend
from matplotlib.patches import Patch
legend = [Patch(color=c, label=l) for l, c in colors.items()]
plt.legend(handles=legend)
plt.title("Sentence Embeddings visualized in 2D (PCA)")
plt.tight_layout()
plt.savefig("practice/embeddings_plot.png")
plt.show()
print("Plot saved to practice/embeddings_plot.png ✓")

# ── PART 4: Semantic search — the full RAG retrieval step ─────────────────────

print("\n--- SEMANTIC SEARCH TEST ---")

test_queries = [
    "I want to work on large scale distributed systems",
    "Tell me about product strategy and metrics",
    "I have experience with data pipelines",
]

for query in test_queries:
    print(f"\nQuery: '{query}'")
    retrieved = retrieve_context(query, k=2)
    for i, chunk in enumerate(retrieved):
        print(f"  [{i+1}] {chunk}")