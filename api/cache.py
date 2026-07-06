
import os
import sys
import json
import time
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
from sentence_transformers import SentenceTransformer
 
# Reuse the same model already loaded by the retriever — no extra memory
_model = None
 
def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model
 
 
CACHE_PATH  = "api/question_cache.json"
CACHE_TTL   = 24 * 60 * 60   # 24 hours in seconds
SIMILARITY_THRESHOLD = 0.92   # cosine similarity above this = cache hit
 
 
# ── Cache I/O ──────────────────────────────────────────────────────────────────
 
def _load_cache() -> list:
    """Load cache from disk. Returns list of cache entries."""
    if not os.path.exists(CACHE_PATH):
        return []
    try:
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return []
 
 
def _save_cache(entries: list):
    """Save cache to disk."""
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(entries, f, indent=2)
 
 
def _purge_expired(entries: list) -> list:
    """Remove entries older than TTL."""
    now = time.time()
    return [e for e in entries if now - e["timestamp"] < CACHE_TTL]
 
 
# ── Similarity ─────────────────────────────────────────────────────────────────
 
def _cosine_similarity(a: list, b: list) -> float:
    """Compute cosine similarity between two embedding vectors."""
    a = np.array(a)
    b = np.array(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
 
 
def _build_cache_key(company: str, role: str, round_type: str, topic: str, difficulty: str) -> str:
    """
    Build a string that captures the semantic meaning of the query.
    This gets embedded and compared against cached queries.
    """
    return f"{company} {role} {round_type} {difficulty} question about {topic}"
 
 
# ── Public API ─────────────────────────────────────────────────────────────────
 
def get_cached_question(company: str, role: str, round_type: str,
                        topic: str, difficulty: str) -> dict | None:
    """
    Check if a semantically similar question was generated recently.
    Returns cached result dict or None if no cache hit.
 
    A cache hit requires:
    1. Cosine similarity > SIMILARITY_THRESHOLD (0.92)
    2. Entry not older than 24 hours
    3. Same round_type (don't mix DSA and Behavioral)
    """
    entries = _load_cache()
    entries = _purge_expired(entries)
 
    if not entries:
        return None
 
    query_key  = _build_cache_key(company, role, round_type, topic, difficulty)
    query_emb  = _get_model().encode(query_key).tolist()
 
    best_score = 0.0
    best_entry = None
 
    for entry in entries:
        # Only match same round_type — don't return a DSA question for a Behavioral query
        if entry.get("round_type") != round_type:
            continue
 
        cached_emb = entry.get("embedding", [])
        if not cached_emb:
            continue
 
        score = _cosine_similarity(query_emb, cached_emb)
        if score > best_score:
            best_score = score
            best_entry = entry
 
    if best_score >= SIMILARITY_THRESHOLD and best_entry:
        print(f"[cache] HIT (similarity={best_score:.3f}) for: {query_key[:60]}")
        return best_entry["result"]
 
    print(f"[cache] MISS (best similarity={best_score:.3f}) for: {query_key[:60]}")
    return None
 
 
def cache_question(company: str, role: str, round_type: str,
                   topic: str, difficulty: str, result: dict):
    """
    Store a generated question in the semantic cache.
    result should be the full dict returned by generate_question_node:
    {"question": "...", "difficulty": "...", "topic": "...", "examples": [...]}
    """
    entries  = _load_cache()
    entries  = _purge_expired(entries)
 
    query_key = _build_cache_key(company, role, round_type, topic, difficulty)
    embedding = _get_model().encode(query_key).tolist()
 
    entry = {
        "timestamp":  time.time(),
        "company":    company,
        "role":       role,
        "round_type": round_type,
        "topic":      topic,
        "difficulty": difficulty,
        "query_key":  query_key,
        "embedding":  embedding,
        "result":     result
    }
 
    entries.append(entry)
    _save_cache(entries)
    print(f"[cache] STORED: {query_key[:60]}")
 
 
def clear_cache():
    """Clear the entire cache. Useful for testing."""
    if os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)
        print("[cache] Cache cleared.")
 
 
def cache_stats() -> dict:
    """Return cache statistics."""
    entries = _load_cache()
    valid   = _purge_expired(entries)
    expired = len(entries) - len(valid)
 
    return {
        "total_entries":   len(entries),
        "valid_entries":   len(valid),
        "expired_entries": expired,
        "companies":       list(set(e["company"]    for e in valid)),
        "round_types":     list(set(e["round_type"] for e in valid)),
    }
 
 
# ── Test ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing semantic cache...\n")
 
    clear_cache()
 
    # Store a question
    fake_result = {
        "question":   "In Amazon's order processing system, given an array of order IDs, find two that sum to a target fraud threshold.",
        "difficulty": "easy",
        "topic":      "arrays",
        "examples":   [{"input": "[1, 5, 3, 7], target=8", "output": "[1, 7]"}]
    }
    cache_question("Amazon", "SWE", "DSA", "arrays", "easy", fake_result)
 
    # Exact match — should hit
    hit = get_cached_question("Amazon", "SWE", "DSA", "arrays", "easy")
    print(f"\nExact match hit: {hit is not None}")
 
    # Similar query — should hit
    hit = get_cached_question("Amazon", "SWE", "DSA", "array manipulation", "easy")
    print(f"Similar query hit: {hit is not None}")
 
    # Different company — probably miss
    hit = get_cached_question("Google", "SWE", "DSA", "arrays", "easy")
    print(f"Different company hit: {hit is not None}")
 
    # Different round — should miss (enforced by round_type filter)
    hit = get_cached_question("Amazon", "SWE", "Behavioral", "arrays", "easy")
    print(f"Different round hit: {hit is not None} (should be False)")
 
    print(f"\nCache stats: {cache_stats()}")