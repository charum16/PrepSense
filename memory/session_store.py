import json
import os
from datetime import datetime

STORE_PATH = "memory/sessions.json"

def _load_store() -> dict:
    if not os.path.exists(STORE_PATH):
        return {}
    with open(STORE_PATH, "r") as f:
        return json.load(f)

def _save_store(data: dict):
    with open(STORE_PATH, "w") as f:
        json.dump(data, f, indent=2)

def _scope_key(user_id: str, company: str, role: str) -> str:
    return f"{user_id}_{company}_{role}"

def _normalize_topic(topic: str) -> str:
    """Normalize topic casing so 'array', 'Array', 'ARRAY' all map to same key."""
    return topic.strip().lower()

def save_session(user_id: str, company: str, role: str, round_type: str, scores: list):
    store = _load_store()
    key   = _scope_key(user_id, company, role)

    if key not in store:
        store[key] = {"sessions": [], "weak_areas": []}

    # Normalize topic casing before saving
    normalized_scores = [
        {**s, "topic": _normalize_topic(s.get("topic", "unknown"))}
        for s in scores
    ]

    session = {
        "timestamp":  datetime.now().isoformat(),
        "company":    company,
        "role":       role,
        "round_type": round_type,
        "scores":     normalized_scores,
        "avg_score":  round(sum(s["score"] for s in normalized_scores) / len(normalized_scores), 1) if normalized_scores else 0
    }

    store[key]["sessions"].append(session)
    store[key]["weak_areas"] = _compute_weak_areas(store[key]["sessions"])
    _save_store(store)

def get_weak_areas(user_id: str, company: str = "", role: str = "") -> list:
    store = _load_store()
    key   = _scope_key(user_id, company, role)
    if key not in store:
        return []
    return store[key].get("weak_areas", [])

def get_session_history(user_id: str, company: str = "", role: str = "") -> list:
    store = _load_store()
    key   = _scope_key(user_id, company, role)
    if key not in store:
        return []
    return store[key].get("sessions", [])

def get_avg_score_by_topic(user_id: str, company: str = "", role: str = "") -> dict:
    store = _load_store()
    key   = _scope_key(user_id, company, role)
    if key not in store:
        return {}

    topic_scores = {}
    for session in store[key]["sessions"]:
        for s in session["scores"]:
            topic = _normalize_topic(s.get("topic", "unknown"))
            topic_scores.setdefault(topic, []).append(s["score"])

    return {
        topic: round(sum(scores) / len(scores), 1)
        for topic, scores in topic_scores.items()
    }

def _compute_weak_areas(sessions: list) -> list:
    topic_scores = {}
    for session in sessions:
        for s in session["scores"]:
            topic = _normalize_topic(s.get("topic", "unknown"))
            topic_scores.setdefault(topic, []).append(s["score"])

    return [
        topic for topic, scores in topic_scores.items()
        if sum(scores) / len(scores) < 6
    ]