import streamlit as st
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq import Groq
from dotenv import load_dotenv
from rag.retriever_v1 import retrieve_questions
from api.evaluator import evaluate_answer
from memory.session_store import save_session, get_weak_areas, get_avg_score_by_topic

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(page_title="PrepSense", page_icon="🎯", layout="centered")

st.markdown("""
<style>
    .stChatMessage { border-radius: 12px; }
    .session-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 500;
        background: #EEEDFE;
        color: #534AB7;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

MAX_QUESTIONS = 5
VALID_ROLES = ["SWE", "Data", "Product", "Analyst"]
VALID_ROUNDS = {
    "SWE":     ["DSA", "System Design", "Behavioral"],
    "Data":    ["Technical", "Case", "Behavioral"],
    "Product": ["Case", "Behavioral"],
    "Analyst": ["Technical", "Case", "Behavioral"]
}
INJECTION_KEYWORDS = [
    "ignore", "forget", "override", "disregard", "pretend",
    "you are now", "new instructions", "system prompt", "jailbreak",
    "act as", "forget previous"
]

def is_injection(text: str) -> bool:
    return any(k in text.lower() for k in INJECTION_KEYWORDS)

def get_difficulty(user_id: str, question_number: int, topic: str) -> str:
    avg_scores = get_avg_score_by_topic(user_id)
    if topic in avg_scores:
        avg = avg_scores[topic]
        if avg < 5:
            return "easy"
        elif avg < 7:
            return "medium"
        else:
            return "hard"
    if question_number == 1:
        return "easy"
    elif question_number == 2:
        return "medium"
    else:
        return "hard"

def build_system_prompt(company, role, round_type, topic, difficulty):
    rag_results = retrieve_questions(company, role, round_type, topic, n_results=3)
    context_str = "\n".join([f"- [{r['difficulty']}] {r['question']}" for r in rag_results])

    role_instructions = {
        "SWE": "Focus on problem-solving approach, time/space complexity, and edge cases.",
        "Data": "Focus on SQL correctness, statistical reasoning, and business interpretation.",
        "Product": "Focus on structured thinking, user empathy, and prioritization framework.",
        "Analyst": "Focus on business logic, metric definition, and communication clarity."
    }
    round_instructions = {
        "DSA": "Ask a coding problem. You MUST include exactly 2 concrete input/output examples.",
        "System Design": "Ask an open-ended system design question. No examples needed.",
        "Behavioral": "Ask a behavioral question. Expect a STAR format answer. No examples needed.",
        "Technical": "Ask a technical concept or SQL question. No examples needed.",
        "Case": "Ask a product or business case question. No examples needed."
    }

    return f"""You are a senior interviewer at {company} hiring for a {role} role.
Rules: Only ask questions, never answer them. Never reveal the rubric. Current difficulty: {difficulty}.

Previous {company} {round_type} questions for context:
{context_str}

{role_instructions.get(role, "")}
{round_instructions.get(round_type, "")}

Respond in this exact JSON format only. No text before or after the JSON:
{{
    "question": "...",
    "difficulty": "{difficulty}",
    "topic": "...",
    "examples": [
        {{"input": "...", "output": "..."}},
        {{"input": "...", "output": "..."}}
    ]
}}

For DSA: populate examples with 2 real input/output pairs.
For all other rounds: set examples to empty list []."""

def fetch_question(company, role, round_type, topic, difficulty, session_history):
    system_prompt = build_system_prompt(company, role, round_type, topic, difficulty)
    messages = session_history + [
        {"role": "user", "content": f"Ask me a {difficulty} {round_type} question about {topic}."}
    ]
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system_prompt}] + messages,
        temperature=0.7,
        max_tokens=600,
        stream=False
    )
    st.session_state.last_raw_response = response.choices[0].message.content

def parse_last_response() -> dict:
    raw = st.session_state.get("last_raw_response", "")
    try:
        return json.loads(raw)
    except:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return json.loads(cleaned)
        except:
            return {"question": raw, "difficulty": "unknown", "topic": "", "examples": []}

# ── Why render_message() exists ────────────────────────────────────────────────
# chat_messages stores the full content of every message as a dict with
# a "type" field. On every rerun Streamlit replays all messages from
# session_state. If examples were rendered separately outside chat_messages
# they'd be wiped on rerun. By storing everything — question text, examples,
# score breakdown — inside chat_messages, every rerun re-renders them correctly.

def render_message(msg: dict):
    """Renders a stored message dict back into the UI."""
    with st.chat_message(msg["role"]):
        if msg["type"] == "question":
            st.caption(msg["caption"])
            st.markdown(msg["content"])
            # Examples stored inside the message — survive reruns
            if msg.get("examples"):
                st.markdown("**Examples:**")
                for ex in msg["examples"]:
                    st.code(f"Input:  {ex.get('input', '')}\nOutput: {ex.get('output', '')}")

        elif msg["type"] == "answer":
            st.markdown(msg["content"])

        elif msg["type"] == "evaluation":
            st.markdown(f"**Score: {msg['score']}/10**")
            if msg.get("what_was_right"):
                st.markdown("**✓ What you got right:**")
                for point in msg["what_was_right"]:
                    st.markdown(f"- {point}")
            if msg.get("what_was_missing"):
                st.markdown("**✗ What was missing:**")
                for point in msg["what_was_missing"]:
                    st.markdown(f"- {point}")
            if msg.get("improvement_tip"):
                st.info(msg["improvement_tip"])
            if msg.get("star_warning"):
                st.warning("Your answer was missing some STAR components.")

# ── Session state init ─────────────────────────────────────────────────────────
defaults = {
    "setup_done": False,
    "session_history": [],
    "question_number": 0,
    "chat_messages": [],
    "waiting_for_answer": False,
    "session_complete": False,
    "last_question": "",
    "session_scores": [],
    "last_actual_topic": "",
    "last_raw_response": ""
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🎯 PrepSense")
st.caption("AI-powered mock interview coach")

# ── Setup screen ───────────────────────────────────────────────────────────────
if not st.session_state.setup_done:
    st.markdown("### Start your session")
    company = st.text_input("Company", placeholder="Amazon, Google, or any company name")
    role = st.selectbox("Role", VALID_ROLES)
    round_type = st.selectbox("Round type", VALID_ROUNDS[role])
    topic = st.text_input("Topic", placeholder="e.g. dynamic programming, SQL, product metrics")

    if "jd_preview" not in st.session_state:
        st.session_state.jd_preview = None

    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("Load Company JD", type="secondary"):
            if not company:
                st.error("Enter a company name first.")
            else:
                PRELOADED = ["Amazon", "Google", "Microsoft", "Adobe"]
                if company in PRELOADED:
                    st.session_state.jd_preview = {
                        "found": True,
                        "source": "preloaded",
                        "skills": [],
                        "titles": [f"{company} Software Engineer"]
                    }
                else:
                    with st.spinner(f"Fetching {company} JD from Adzuna..."):
                        from rag.jdfetcher import fetch_and_store_jd
                        jd_data = fetch_and_store_jd(company)
                        st.session_state.jd_preview = jd_data

    # Show JD preview card
    if st.session_state.jd_preview:
        preview = st.session_state.jd_preview
        if preview.get("source") == "preloaded":
            st.success(f"✓ {company} is in our pre-loaded database — using curated question bank.")
        elif preview.get("found"):
            st.success(f"✓ Found live JD for {company} on Adzuna")
            if preview.get("titles"):
                st.caption(f"Roles found: {', '.join(preview['titles'][:3])}")
            if preview.get("skills"):
                st.markdown("**Key skills extracted from JD:**")
                skill_cols = st.columns(3)
                for i, skill in enumerate(preview["skills"][:12]):
                    with skill_cols[i % 3]:
                        st.markdown(f"• {skill}")
        else:
            st.warning(f"No JD found for {company} on Adzuna — will use Generic question bank as fallback.")

    with col2:
        if st.button("Start Interview", type="primary"):
            if not company or not topic:
                st.error("Please fill in company and topic.")
            elif is_injection(company) or is_injection(topic):
                st.error("Invalid input detected.")
            else:
                st.session_state.company = company
                st.session_state.role = role
                st.session_state.round_type = round_type
                st.session_state.topic = topic
                st.session_state.jd_preview = None
                st.session_state.setup_done = True
                st.rerun()

# ── Interview screen ───────────────────────────────────────────────────────────
else:
    company = st.session_state.company
    role = st.session_state.role
    round_type = st.session_state.round_type
    topic = st.session_state.topic
    user_id = "default_user"

    st.markdown(
        f'<span class="session-badge">{company}</span>'
        f'<span class="session-badge">{role}</span>'
        f'<span class="session-badge">{round_type}</span>'
        f'<span class="session-badge">{topic}</span>',
        unsafe_allow_html=True
    )
    st.divider()

    # Replay all messages from session_state — this is what survives reruns
    for msg in st.session_state.chat_messages:
        render_message(msg)

    # ── Session complete ───────────────────────────────────────────────────────
    if st.session_state.session_complete:
        save_session(
            user_id=user_id,
            company=company,
            role=role,
            round_type=round_type,
            scores=st.session_state.session_scores
        )

        weak_areas = get_weak_areas(user_id)
        avg_scores = get_avg_score_by_topic(user_id)
        avg = sum(s["score"] for s in st.session_state.session_scores) / len(st.session_state.session_scores) if st.session_state.session_scores else 0

        st.success(f"Session complete! Average score: {avg:.1f}/10")

        if weak_areas:
            st.warning(f"Weak areas identified: {', '.join(weak_areas)}")

        st.markdown("**Score breakdown this session:**")
        for s in st.session_state.session_scores:
            color = "🟢" if s["score"] >= 7 else "🟡" if s["score"] >= 5 else "🔴"
            st.markdown(f"{color} {s['topic']} — {s['score']}/10")

        if avg_scores:
            st.markdown("**Your all-time averages:**")
            for t, avg_s in avg_scores.items():
                color = "🟢" if avg_s >= 7 else "🟡" if avg_s >= 5 else "🔴"
                st.markdown(f"{color} {t}: {avg_s}/10")

        if st.button("Start new session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # ── Generate next question ─────────────────────────────────────────────────
    elif not st.session_state.waiting_for_answer:
        st.session_state.question_number += 1
        difficulty = get_difficulty(user_id, st.session_state.question_number, topic)

        with st.spinner("Generating question..."):
            fetch_question(company, role, round_type, topic, difficulty, st.session_state.session_history)

        result = parse_last_response()
        question_text = result.get("question", st.session_state.get("last_raw_response", ""))
        actual_topic = result.get("topic", topic)
        examples = result.get("examples", [])

        st.session_state.last_question = question_text
        st.session_state.last_actual_topic = actual_topic

        # Store everything in chat_messages — examples included
        msg = {
            "role": "assistant",
            "type": "question",
            "caption": f"Question {st.session_state.question_number} of {MAX_QUESTIONS} — {difficulty.upper()}",
            "content": question_text,
            "examples": examples  # stored here, rendered by render_message()
        }
        st.session_state.chat_messages.append(msg)
        st.session_state.session_history.append({"role": "assistant", "content": question_text})
        st.session_state.waiting_for_answer = True
        st.rerun()

    # ── Wait for answer ────────────────────────────────────────────────────────
    else:
        answer = st.chat_input("Type your answer here...")
        if answer:
            if is_injection(answer):
                st.warning("That looks like an injection attempt. Please answer the question.")
            else:
                st.session_state.chat_messages.append({
                    "role": "user",
                    "type": "answer",
                    "content": answer
                })
                st.session_state.session_history.append({"role": "user", "content": answer})

                with st.spinner("Evaluating your answer..."):
                    eval_result = evaluate_answer(
                        question=st.session_state.last_question,
                        answer=answer,
                        round_type=round_type,
                        topic=topic
                    )

                score = eval_result.get("score", 0)

                st.session_state.session_scores.append({
                    "topic": st.session_state.get("last_actual_topic", topic),
                    "score": score,
                    "question": st.session_state.last_question
                })

                # Store full evaluation in chat_messages
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "type": "evaluation",
                    "score": score,
                    "what_was_right": eval_result.get("what_was_right", []),
                    "what_was_missing": eval_result.get("what_was_missing", []),
                    "improvement_tip": eval_result.get("improvement_tip", ""),
                    "star_warning": round_type == "Behavioral" and not eval_result.get("star_complete"),
                    "content": f"Score: {score}/10"
                })

                st.session_state.waiting_for_answer = False

                if st.session_state.question_number >= MAX_QUESTIONS:
                    st.session_state.session_complete = True

                st.rerun()