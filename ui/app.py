import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from agents.interview_graph import build_question_graph, build_answer_graph, create_initial_state
from agents.study_plan_agent import generate_study_plan
from memory.session_store import save_session, get_weak_areas, get_avg_score_by_topic

load_dotenv()

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


@st.cache_resource
def get_question_graph():
    return build_question_graph()


@st.cache_resource
def get_answer_graph():
    return build_answer_graph()


# ── render_message ─────────────────────────────────────────────────────────────
def render_message(msg: dict):
    with st.chat_message(msg["role"]):
        if msg["type"] == "question":
            st.caption(msg["caption"])
            st.markdown(msg["content"])
            if msg.get("examples"):
                st.markdown("**Examples:**")
                for ex in msg["examples"]:
                    inp = ex.get("input", "")
                    out = ex.get("output", "")
                    # Pretty-print dicts/lists as JSON
                    if isinstance(inp, (dict, list)):
                        import json as _json
                        inp = _json.dumps(inp, indent=2)
                    if isinstance(out, (dict, list)):
                        import json as _json
                        out = _json.dumps(out, indent=2)
                    st.code(f"Input:  {inp}\nOutput: {out}", language="json")

        elif msg["type"] == "answer":
            st.markdown(msg["content"])

        elif msg["type"] == "evaluation":
            score = msg["score"]
            icon  = "🟢" if score >= 7 else "🟡" if score >= 5 else "🔴"
            st.markdown(f"{icon} **Score: {score}/10**")
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


# ── render_study_plan ──────────────────────────────────────────────────────────
def render_study_plan(plan: dict):
    company = plan.get("target_company", "")
    role    = plan.get("target_role", "")
    hours   = plan.get("total_hours", 35)

    st.markdown(f"### 📅 7-Day Study Plan — {company} · {role}")
    st.caption(f"~{hours} hours total · 4-5 hours/day · Personalized to your weak areas")

    if plan.get("weak_areas"):
        st.warning(f"⚠ Priority focus: **{', '.join(plan['weak_areas'])}**")

    if plan.get("final_tip"):
        st.info(f"💡 **Pro tip:** {plan['final_tip']}")

    days = plan.get("days", [])
    if not days:
        st.warning("Could not generate study plan. Complete more sessions to build your profile.")
        return

    for day in days:
        day_num        = day.get("day", "")
        focus          = day.get("focus", "")
        day_hours      = day.get("hours", 5)
        schedule       = day.get("schedule", [])
        tasks          = day.get("tasks", [])
        resources      = day.get("resources", [])
        success_metric = day.get("success_metric", "")

        with st.expander(f"**Day {day_num}** — {focus} · _{day_hours}h_", expanded=(day_num == 1)):

            if schedule:
                st.markdown("**🕐 Hour-by-Hour Schedule**")
                for slot in schedule:
                    st.markdown(f"- **{slot.get('hour', '')}** — {slot.get('activity', '')}")
                st.divider()

            col1, col2 = st.columns(2)
            with col1:
                if tasks:
                    st.markdown("**✅ Tasks**")
                    for t in tasks:
                        st.markdown(f"- {t}")
            with col2:
                if resources:
                    st.markdown("**📚 Resources**")
                    for r in resources:
                        st.markdown(f"- {r}")

            if success_metric:
                st.success(f"**✓ Done when:** {success_metric}")


# ── Session state defaults ─────────────────────────────────────────────────────
defaults = {
    "setup_done":         False,
    "chat_messages":      [],
    "session_complete":   False,
    "graph_state":        None,
    "waiting_for_answer": False,
    "study_plan":         None,
    "show_study_plan":    False,
    "jd_preview":         None,
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

    company    = st.text_input("Company", placeholder="Amazon, Google, or any company name")
    role       = st.selectbox("Role", VALID_ROLES)
    round_type = st.selectbox("Round type", VALID_ROUNDS[role])
    topic      = st.text_input("Topic", placeholder="e.g. dynamic programming, SQL, product metrics")

    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("Load Company JD", type="secondary"):
            if not company:
                st.error("Enter a company name first.")
            else:
                PRELOADED = ["Amazon", "Google", "Microsoft", "Adobe"]
                if company in PRELOADED:
                    st.session_state.jd_preview = {
                        "found": True, "source": "preloaded",
                        "skills": [], "titles": [f"{company} Software Engineer"]
                    }
                else:
                    with st.spinner(f"Fetching {company} JD from Adzuna..."):
                        from rag.jdfetcher import fetch_and_store_jd
                        st.session_state.jd_preview = fetch_and_store_jd(company)

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
                cols = st.columns(3)
                for i, skill in enumerate(preview["skills"][:12]):
                    with cols[i % 3]:
                        st.markdown(f"• {skill}")
        else:
            st.warning(f"No JD found for {company} — will use Generic question bank.")

    with col2:
        if st.button("Start Interview →", type="primary"):
            if not company or not topic:
                st.error("Please fill in company and topic.")
            elif is_injection(company) or is_injection(topic):
                st.error("Invalid input detected.")
            else:
                st.session_state.graph_state        = create_initial_state(
                    company=company, role=role,
                    round_type=round_type, topic=topic,
                    user_id="default_user"
                )
                st.session_state.jd_preview         = None
                st.session_state.setup_done         = True
                st.session_state.waiting_for_answer = False
                st.session_state.study_plan         = None
                st.session_state.show_study_plan    = False
                st.rerun()

# ── Interview screen ───────────────────────────────────────────────────────────
else:
    state      = st.session_state.graph_state
    company    = state["company"]
    role       = state["role"]
    round_type = state["round_type"]
    topic      = state["topic"]
    user_id    = state["user_id"]
    q_num      = state.get("question_number", 0)
    max_q      = state.get("max_questions", MAX_QUESTIONS)

    st.markdown(
        f'<span class="session-badge">{company}</span>'
        f'<span class="session-badge">{role}</span>'
        f'<span class="session-badge">{round_type}</span>'
        f'<span class="session-badge">{topic}</span>',
        unsafe_allow_html=True
    )

    if not st.session_state.session_complete:
        st.progress(q_num / max_q if max_q > 0 else 0,
                    text=f"Question {q_num} of {max_q}")

    st.divider()

    for msg in st.session_state.chat_messages:
        render_message(msg)

    # ── Session complete ───────────────────────────────────────────────────────
    if st.session_state.session_complete:
        scores = state["session_scores"]
        avg    = sum(s["score"] for s in scores) / len(scores) if scores else 0

        save_session(
            user_id=user_id,
            company=company,
            role=role,
            round_type=round_type,
            scores=scores
        )

        # Try calling get_weak_areas with company+role; fall back if not supported
        try:
            weak_areas = get_weak_areas(user_id, company, role)
            avg_scores = get_avg_score_by_topic(user_id, company, role)
        except TypeError:
            weak_areas = get_weak_areas(user_id)
            avg_scores = get_avg_score_by_topic(user_id)

        icon = "🟢" if avg >= 7 else "🟡" if avg >= 5 else "🔴"
        st.success(f"Session complete! {icon} Average score: **{avg:.1f}/10**")

        if weak_areas:
            st.warning(f"⚠ Weak areas: **{', '.join(weak_areas)}**")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**This session:**")
            for s in scores:
                c = "🟢" if s["score"] >= 7 else "🟡" if s["score"] >= 5 else "🔴"
                st.markdown(f"{c} {s['topic']} — **{s['score']}/10**")
        with col2:
            if avg_scores:
                st.markdown(f"**All-time — {company} · {role}:**")
                for t, avg_s in avg_scores.items():
                    c = "🟢" if avg_s >= 7 else "🟡" if avg_s >= 5 else "🔴"
                    st.markdown(f"{c} {t}: **{avg_s}/10**")

        st.divider()

        if not st.session_state.show_study_plan:
            c1, c2 = st.columns(2)
            with c1:
                if st.button("📅 Generate My 7-Day Study Plan", type="primary", use_container_width=True):
                    with st.spinner("Generating your personalized study plan..."):
                        st.session_state.study_plan = generate_study_plan(
                            user_id, company, role,
                            session_scores=state["session_scores"]
                        )
                        st.session_state.show_study_plan = True
                    st.rerun()
            with c2:
                if st.button("🔄 Start new session", use_container_width=True):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()
        else:
            render_study_plan(st.session_state.study_plan)
            st.divider()
            if st.button("🔄 Start new session"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

    # ── Generate next question ─────────────────────────────────────────────────
    elif not st.session_state.waiting_for_answer:
        with st.spinner(f"Generating question {q_num + 1} of {max_q}..."):
            state = get_question_graph().invoke(state)

        st.session_state.graph_state = state

        question_text = state["current_question"]
        q_num         = state["question_number"]
        difficulty    = state["current_difficulty"]
        examples      = state.get("current_examples", [])

        st.session_state.chat_messages.append({
            "role":     "assistant",
            "type":     "question",
            "caption":  f"Question {q_num} of {state['max_questions']} — {difficulty.upper()}",
            "content":  question_text,
            "examples": examples       # always passed; render_message shows them if non-empty
        })

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
                    "role": "user", "type": "answer", "content": answer
                })

                state["user_answer"] = answer

                with st.spinner("Evaluating..."):
                    state = get_answer_graph().invoke(state)

                st.session_state.graph_state = state

                # Clarification
                if state.get("clarification_response"):
                    st.session_state.chat_messages.append({
                        "role":     "assistant",
                        "type":     "question",
                        "caption":  "Interviewer clarification",
                        "content":  state["clarification_response"],
                        "examples": []
                    })
                    st.session_state.graph_state["clarification_response"] = ""
                    st.session_state.waiting_for_answer = True
                    st.rerun()

                # Evaluation
                else:
                    eval_result = state.get("last_evaluation", {})
                    score       = state.get("current_score", 0)

                    st.session_state.chat_messages.append({
                        "role":             "assistant",
                        "type":             "evaluation",
                        "score":            score,
                        "what_was_right":   eval_result.get("what_was_right", []),
                        "what_was_missing": eval_result.get("what_was_missing", []),
                        "improvement_tip":  eval_result.get("improvement_tip", ""),
                        "star_warning": (
                            round_type == "Behavioral"
                            and not eval_result.get("star_complete", True)
                        ),
                        "content": f"Score: {score}/10"
                    })

                    if state["question_number"] >= state["max_questions"]:
                        st.session_state.session_complete = True
                    else:
                        st.session_state.waiting_for_answer = False

                    st.rerun()