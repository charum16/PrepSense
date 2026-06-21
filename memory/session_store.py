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

def save_session(user_id: str, company: str, role: str, round_type: str, scores: list[dict]):
    store = _load_store()
    if user_id not in store:
        store[user_id] = {"sessions": [], "weak_areas": []}

    session = {
        "timestamp": datetime.now().isoformat(),
        "company": company,
        "role": role,
        "round_type": round_type,
        "scores": scores,
        "avg_score": round(sum(s["score"] for s in scores) / len(scores), 1) if scores else 0
    }

    store[user_id]["sessions"].append(session)
    store[user_id]["weak_areas"] = _compute_weak_areas(store[user_id]["sessions"])
    _save_store(store)

def get_weak_areas(user_id: str) -> list:
    store = _load_store()
    if user_id not in store:
        return []
    return store[user_id].get("weak_areas", [])

def get_session_history(user_id: str) -> list:
    store = _load_store()
    if user_id not in store:
        return []
    return store[user_id].get("sessions", [])

def get_avg_score_by_topic(user_id: str) -> dict:
    store = _load_store()
    if user_id not in store:
        return {}
    topic_scores = {}
    for session in store[user_id]["sessions"]:
        for s in session["scores"]:
            topic = s.get("topic", "unknown")
            if topic not in topic_scores:
                topic_scores[topic] = []
            topic_scores[topic].append(s["score"])
    return {topic: round(sum(scores) / len(scores), 1) for topic, scores in topic_scores.items()}

def _compute_weak_areas(sessions: list) -> list:
    topic_scores = {}
    for session in sessions:
        for s in session["scores"]:
            topic = s.get("topic", "unknown")
            if topic not in topic_scores:
                topic_scores[topic] = []
            topic_scores[topic].append(s["score"])
    return [
        topic for topic, scores in topic_scores.items()
        if sum(scores) / len(scores) < 6
    ]