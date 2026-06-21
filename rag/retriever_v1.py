
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_store")


def _get_collection(name: str):
    """Safely get a collection, returns None if it doesn't exist."""
    try:
        return client.get_collection(name)
    except:
        return None


def retrieve_from_preloaded(company: str, role: str, round_type: str, topic: str, n_results: int) -> list:
    """
    Query the pre-loaded interview_questions collection.
    Filters by company + role + round for precise retrieval.
    """
    collection = _get_collection("interview_questions")
    if not collection:
        return []

    query_embedding = model.encode(topic).tolist()

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={
                "$and": [
                    {"company": {"$eq": company}},
                    {"role": {"$eq": role}},
                    {"round": {"$eq": round_type}}
                ]
            }
        )
        questions = results["documents"][0]
        metadatas = results["metadatas"][0]

        return [
            {
                "question": questions[i],
                "topic": metadatas[i]["topic"],
                "difficulty": metadatas[i]["difficulty"],
                "round": metadatas[i]["round"],
                "source": "preloaded"
            }
            for i in range(len(questions))
        ]
    except:
        return []


def retrieve_from_fetched_jd(company: str, topic: str, n_results: int) -> list:
    """
    Query the fetched_jds collection for a company.
    Returns raw JD paragraphs as context — no role/round filtering
    because fetched JDs don't have that metadata.
    WHY return raw paragraphs not questions?
    These aren't questions — they're JD context chunks.
    The question generator uses them as background context to
    generate relevant questions, same as it uses pre-loaded questions.
    """
    collection = _get_collection("fetched_jds")
    if not collection:
        return []

    query_embedding = model.encode(topic).tolist()

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"company": {"$eq": company}}
        )
        chunks = results["documents"][0]

        return [
            {
                "question": chunk,   
                "topic": topic,
                "difficulty": "medium",
                "round": "General",
                "source": "fetched_jd"
            }
            for chunk in chunks
        ]
    except:
        return []


def retrieve_questions(company: str, role: str, round_type: str, topic: str, n_results: int = 3) -> list:
    """
    Main retrieval function — three-tier fallback:
    1. Pre-loaded company data (best — structured, role/round filtered)
    2. Fetched JD for this company (good — real JD, no structure)
    3. Generic pre-loaded data (fallback — always works)
    """
    # Tier 1 — pre-loaded company-specific data
    results = retrieve_from_preloaded(company, role, round_type, topic, n_results)
    if results:
        print(f"[retriever] Using pre-loaded data for {company}")
        return results

    # Tier 2 — live fetched JD for this company
    results = retrieve_from_fetched_jd(company, topic, n_results)
    if results:
        print(f"[retriever] Using fetched JD for {company}")
        return results

    # Tier 3 — Generic fallback
    print(f"[retriever] No data for {company} — falling back to Generic")
    results = retrieve_from_preloaded("Generic", role, round_type, topic, n_results)
    return results