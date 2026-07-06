from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sys
import os
import uuid

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.retriever_v1 import retrieve_questions
from api.evaluator import evaluate_answer
from agents.study_plan_agent import generate_study_plan
from memory.session_store import save_session, get_weak_areas, get_avg_score_by_topic
from api.guardrails import check_injection, safe_evaluate

app = FastAPI(
    title="PrepSense API",
    description="AI-powered mock interview coach backend",
    version="1.0.0"
)

# ── CORS — allows Streamlit to call this API ───────────────────────────────────
# WHY CORS?
# Streamlit runs on port 8501, FastAPI on 8000. Browsers block
# cross-origin requests by default. CORSMiddleware tells the browser
# "yes, requests from Streamlit are allowed."
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store ────────────────────────────────────────────────────
# WHY in-memory not database?
# For a demo with one user, RAM is fine and instant.
# If two users run simultaneously, sessions are isolated by session_id.
# Postgres upgrade would just replace this dict — same interface.
SESSIONS: dict = {}


# ── Request/Response models ────────────────────────────────────────────────────
# WHY Pydantic models?
# FastAPI validates every incoming request against these models automatically.
# If a required field is missing or wrong type, FastAPI returns a 422 error
# with a clear message before your code even runs.
# Without Pydantic you'd write manual if/else validation for every field.

class StartSessionRequest(BaseModel):
    company: str
    role: str
    round_type: str
    topic: str
    user_id: Optional[str] = "default_user"

class StartSessionResponse(BaseModel):
    session_id: str
    message: str

class NextQuestionRequest(BaseModel):
    session_id: str

class NextQuestionResponse(BaseModel):
    question: str
    difficulty: str
    topic: str
    examples: list
    question_number: int
    max_questions: int
    session_complete: bool

class SubmitAnswerRequest(BaseModel):
    session_id: str
    answer: str

class SubmitAnswerResponse(BaseModel):
    score: int
    what_was_right: list
    what_was_missing: list
    improvement_tip: str
    star_complete: bool
    session_complete: bool
    avg_score: Optional[float] = None

class StudyPlanRequest(BaseModel):
    user_id: str
    company: str
    role: str

class SessionSummaryResponse(BaseModel):
    avg_score: float
    scores: list
    weak_areas: list
    company: str
    role: str
    round_type: str


# ── Helper: difficulty from session scores ─────────────────────────────────────
def _get_difficulty(session: dict) -> str:
    scores = session.get("scores", [])
    if not scores:
        q_num = session.get("question_number", 1)
        if q_num == 1: return "easy"
        elif q_num == 2: return "medium"
        else: return "hard"
    avg = sum(s["score"] for s in scores) / len(scores)
    if avg < 4: return "easy"
    elif avg < 6.5: return "medium"
    else: return "hard"


# ── Helper: generate question via Groq ────────────────────────────────────────
def _generate_question(session: dict) -> dict:
    from groq import Groq
    from dotenv import load_dotenv
    import json
    load_dotenv()

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    difficulty = _get_difficulty(session)
    company = session["company"]
    role = session["role"]
    round_type = session["round_type"]
    topic = session["topic"]

    rag_results = retrieve_questions(company, role, round_type, topic, n_results=3)
    context_str = "\n".join([f"- [{r['difficulty']}] {r['question']}" for r in rag_results])

    asked = session.get("asked_questions", [])
    asked_str = ""
    if asked:
        asked_str = "\n\nDo NOT repeat these already asked questions:\n" + "\n".join([f"- {q}" for q in asked])

    role_instructions = {
        "SWE": "Focus on problem-solving approach, time/space complexity, and edge cases.",
        "Data": "Focus on SQL correctness, statistical reasoning, and business interpretation.",
        "Product": "Focus on structured thinking, user empathy, and prioritization framework.",
        "Analyst": "Focus on business logic, metric definition, and communication clarity."
    }
    round_instructions = {
        "DSA": "Ask a coding problem. Include exactly 2 concrete input/output examples.",
        "System Design": "Ask an open-ended system design question. No examples needed.",
        "Behavioral": "Ask a behavioral question. Expect STAR format. No examples needed.",
        "Technical": "Ask a technical concept or SQL question. No examples needed.",
        "Case": "Ask a product or business case question. No examples needed."
    }
    difficulty_bar = {
        "easy": "Straightforward concept check. Solvable in under 10 minutes.",
        "medium": "Requires real problem-solving. Optimal solution uses a non-obvious technique.",
        "hard": "Challenging. Requires deep knowledge, optimization, or complex reasoning."
    }

    system = f"""You are a senior interviewer at {company} hiring for {role}.

DIFFICULTY: {difficulty.upper()} — {difficulty_bar.get(difficulty, "")}
Do NOT ask trivial questions like sorting an array or reversing a string for medium/hard.

REAL {company.upper()} QUESTIONS FOR CONTEXT (match this level and style):
{context_str}
{asked_str}

{role_instructions.get(role, "")}
{round_instructions.get(round_type, "")}

Respond in JSON only:
{{
    "question": "...",
    "difficulty": "{difficulty}",
    "topic": "...",
    "examples": []
}}
For DSA populate examples with 2 real input/output pairs. All others keep examples empty."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Ask a {difficulty} {round_type} question about {topic} at {company}."}
        ],
        temperature=0.8,
        max_tokens=600,
        stream=False
    )

    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return json.loads(cleaned)
        except:
            return {"question": raw, "difficulty": difficulty, "topic": topic, "examples": []}


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "PrepSense API"}


@app.post("/start-session", response_model=StartSessionResponse)
def start_session(req: StartSessionRequest):
    """
    Creates a new interview session.
    Returns a session_id used for all subsequent calls.
    """
    # Validate inputs
    flagged, reason = check_injection(req.company + req.topic)
    if flagged:
        raise HTTPException(status_code=400, detail=f"Invalid input: {reason}")

    valid_roles = ["SWE", "Data", "Product", "Analyst"]
    if req.role not in valid_roles:
        raise HTTPException(status_code=400, detail=f"Invalid role. Choose from {valid_roles}")

    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {
        "company": req.company,
        "role": req.role,
        "round_type": req.round_type,
        "topic": req.topic,
        "user_id": req.user_id,
        "question_number": 0,
        "max_questions": 5,
        "scores": [],
        "asked_questions": [],
        "current_question": "",
        "complete": False
    }

    return StartSessionResponse(
        session_id=session_id,
        message=f"Session started for {req.company} {req.role} — {req.round_type} on {req.topic}"
    )


@app.post("/next-question", response_model=NextQuestionResponse)
def next_question(req: NextQuestionRequest):
    """
    Generates the next interview question for the session.
    Automatically escalates difficulty based on running score average.
    """
    session = SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session["complete"]:
        raise HTTPException(status_code=400, detail="Session already complete")

    session["question_number"] += 1
    result = _generate_question(session)

    question_text = result.get("question", "")
    session["current_question"] = question_text
    session["asked_questions"].append(question_text)

    return NextQuestionResponse(
        question=question_text,
        difficulty=result.get("difficulty", "medium"),
        topic=result.get("topic", session["topic"]),
        examples=result.get("examples", []),
        question_number=session["question_number"],
        max_questions=session["max_questions"],
        session_complete=False
    )


@app.post("/submit-answer", response_model=SubmitAnswerResponse)
def submit_answer(req: SubmitAnswerRequest):
    """
    Evaluates a user's answer using LLM-as-judge.
    Updates session scores and determines if session is complete.
    """
    session = SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Injection check on answer
    flagged, reason = check_injection(req.answer)
    if flagged:
        raise HTTPException(status_code=400, detail=f"Invalid input: {reason}")

    # Evaluate with guardrails
    eval_result = safe_evaluate(
        evaluate_fn=evaluate_answer,
        question=session["current_question"],
        answer=req.answer,
        round_type=session["round_type"],
        topic=session["topic"]
    )

    score = eval_result.get("score", 0)
    session["scores"].append({
        "topic": session["topic"],
        "score": score,
        "question": session["current_question"]
    })

    # Check completion
    session_complete = session["question_number"] >= session["max_questions"]
    if session_complete:
        session["complete"] = True
        # Persist to JSON store
        save_session(
            user_id=session["user_id"],
            company=session["company"],
            role=session["role"],
            round_type=session["round_type"],
            scores=session["scores"]
        )

    avg = sum(s["score"] for s in session["scores"]) / len(session["scores"])

    return SubmitAnswerResponse(
        score=score,
        what_was_right=eval_result.get("what_was_right", []),
        what_was_missing=eval_result.get("what_was_missing", []),
        improvement_tip=eval_result.get("improvement_tip", ""),
        star_complete=eval_result.get("star_complete", False),
        session_complete=session_complete,
        avg_score=round(avg, 1)
    )


@app.post("/study-plan")
def study_plan(req: StudyPlanRequest):
    """
    Generates a personalized 7-day study plan based on
    the user's weak areas from past sessions.
    """
    plan = generate_study_plan(
        user_id=req.user_id,
        company=req.company,
        role=req.role
    )
    return plan


@app.get("/session-summary/{session_id}", response_model=SessionSummaryResponse)
def session_summary(session_id: str):
    """
    Returns the summary of a completed session.
    """
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    scores = session.get("scores", [])
    avg = sum(s["score"] for s in scores) / len(scores) if scores else 0
    weak_areas = get_weak_areas(session["user_id"], session["company"], session["role"])

    return SessionSummaryResponse(
        avg_score=round(avg, 1),
        scores=scores,
        weak_areas=weak_areas,
        company=session["company"],
        role=session["role"],
        round_type=session["round_type"]
    )


@app.get("/weak-areas/{user_id}")
def weak_areas(user_id: str, company: str = "", role: str = ""):
    return {
        "user_id": user_id,
        "weak_areas": get_weak_areas(user_id, company, role),
        "avg_scores": get_avg_score_by_topic(user_id, company, role)
    }