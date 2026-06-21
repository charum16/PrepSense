import json
import chromadb
from sentence_transformers import SentenceTransformer

with open("data/prev_year_qs.json", "r") as f:
    data = json.load(f)

client = chromadb.PersistentClient(path="./chroma_store")

try:
    client.delete_collection("interview_questions")
except:
    pass

collection = client.create_collection("interview_questions")

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = []
metadatas = []
ids = []

for company, roles in data.items():
    for role, rounds in roles.items():
        for round_type, questions in rounds.items():
            for i, q in enumerate(questions):
                documents.append(q["question"])
                metadatas.append({
                    "company": company,
                    "role": role,
                    "round": round_type,
                    "topic": q["topic"],
                    "difficulty": q["difficulty"]
                })
                ids.append(f"{company}_{role}_{round_type}_{i}")

embeddings = model.encode(documents).tolist()

collection.add(
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids
)

print(f"Loaded {len(documents)} questions into ChromaDB")
print("Companies:", list(data.keys()))