import os
import json
import requests
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv

load_dotenv()

ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY")

model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_store")



def fetch_jd(company: str) -> dict:
    """
    Fetch top job descriptions for a company from Adzuna.
    Returns: {
        "company": str,
        "raw_text": str,        # merged JD text
        "skills": list[str],    # extracted key skills
        "titles": list[str],    # job titles found
        "found": bool
    }
    """
    url = f"https://api.adzuna.com/v1/api/jobs/us/search/1"
    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_APP_KEY,
        "what": f"{company} software engineer",
        "results_per_page": 3,
        "content-type": "application/json"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])
        if not results:
            return {"company": company, "raw_text": "", "skills": [], "titles": [], "found": False}

        # Merge top 3 JD descriptions into one corpus
        merged_text = ""
        titles = []
        for job in results:
            title = job.get("title", "")
            description = job.get("description", "")
            if title:
                titles.append(title)
            if description:
                merged_text += f"\n\nRole: {title}\n{description}"

        # Extract skills using simple LLM call
        skills = extract_skills(merged_text, company)

        return {
            "company": company,
            "raw_text": merged_text.strip(),
            "skills": skills,
            "titles": titles,
            "found": bool(merged_text.strip())
        }

    except requests.exceptions.RequestException as e:
        print(f"Adzuna API error: {e}")
        return {"company": company, "raw_text": "", "skills": [], "titles": [], "found": False}



def extract_skills(jd_text: str, company: str) -> list:
    """Use Groq to extract key skills from raw JD text."""
    try:
        from groq import Groq
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        prompt = f"""Extract the key technical skills, tools, and competencies from this job description for {company}.

JD Text:
{jd_text[:3000]}

Return a JSON array of strings only. Max 15 items. Focus on specific technologies, frameworks, and skills.
Example: ["Python", "distributed systems", "SQL", "system design", "AWS", "data structures"]

Return only the JSON array, nothing else."""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300
        )

        raw = response.choices[0].message.content.strip()
        cleaned = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(cleaned)

    except Exception as e:
        print(f"Skill extraction failed: {e}")
        return []



def chunk_and_store(jd_data: dict) -> int:
    """
    Chunks JD text, embeds it, stores in ChromaDB.
    Returns number of chunks stored.
    """
    company = jd_data["company"]
    raw_text = jd_data["raw_text"]

    if not raw_text:
        return 0

    # Chunk by double newline, keep chunks > 80 chars
    chunks = [c.strip() for c in raw_text.split("\n\n") if len(c.strip()) > 80]

    if not chunks:
        return 0

    try:
        collection = chroma_client.get_collection("fetched_jds")
    except:
        collection = chroma_client.create_collection("fetched_jds")

    # Remove old entries for this company (fresh fetch = fresh data)
    try:
        existing = collection.get(where={"company": company})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except:
        pass

    # Embed and store
    embeddings = model.encode(chunks).tolist()
    ids = [f"{company}_jd_{i}" for i in range(len(chunks))]
    metadatas = [{"company": company, "role": "General", "round": "General"} for _ in chunks]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    return len(chunks)


# ── Step 4: Main entry point — fetch + store in one call ──────────────────────
def fetch_and_store_jd(company: str) -> dict:
    """
    Full pipeline: fetch JD → extract skills → store in ChromaDB.
    Called by ui/app.py when user types a company name.
    Returns jd_data dict with found, skills, titles for UI preview.
    """
    print(f"Fetching JD for {company}...")
    jd_data = fetch_jd(company)

    if jd_data["found"]:
        chunks_stored = chunk_and_store(jd_data)
        print(f"Stored {chunks_stored} chunks for {company} ✓")
    else:
        print(f"No JD found for {company} — will use Generic fallback")

    return jd_data


