import streamlit as st
import sys
import os
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from agents.interview_graph import build_question_graph, build_answer_graph, create_initial_state
from agents.study_plan_agent import generate_study_plan
from memory.session_store import save_session, get_weak_areas, get_avg_score_by_topic, get_session_history
from api.guardrails import check_injection

load_dotenv()

st.set_page_config(
    page_title="PrepSense",
    page_icon="◆",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0D1117 !important;
    color: #E6EDF3 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

[data-testid="stAppViewContainer"] > .main { background: #0D1117 !important; }
[data-testid="stHeader"] { background: #0D1117 !important; border-bottom: 1px solid #21262D; }
section[data-testid="stSidebar"] { background: #161B22 !important; }

h1, h2, h3 { color: #E6EDF3 !important; letter-spacing: -0.02em; }

#MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }
.stDeployButton { display: none !important; }

.ps-wordmark { display: flex; align-items: center; gap: 10px; padding: 28px 0 6px; }
.ps-diamond { width: 28px; height: 28px; background: #6E56CF; transform: rotate(45deg); border-radius: 4px; flex-shrink: 0; }
.ps-name { font-size: 18px; font-weight: 600; color: #E6EDF3; letter-spacing: -0.03em; }
.ps-tagline { font-size: 12px; color: #7D8590; margin-top: 2px; letter-spacing: 0.01em; }

.badge-row { display: flex; gap: 6px; flex-wrap: wrap; margin: 16px 0 0; }
.badge { font-size: 11px; font-weight: 500; padding: 3px 10px; border-radius: 4px; background: #21262D; color: #7D8590; border: 1px solid #30363D; letter-spacing: 0.02em; text-transform: uppercase; }
.badge.active { background: rgba(110, 86, 207, 0.15); color: #A78BFA; border-color: rgba(110, 86, 207, 0.4); }

.ps-divider { height: 1px; background: #21262D; margin: 20px 0; }

.progress-wrap { margin: 12px 0 20px; }
.progress-label { font-size: 11px; color: #7D8590; margin-bottom: 6px; letter-spacing: 0.04em; text-transform: uppercase; }
.progress-track { height: 2px; background: #21262D; border-radius: 2px; overflow: hidden; }
.progress-fill { height: 100%; background: #6E56CF; border-radius: 2px; transition: width 0.4s ease; }

.q-card { background: #161B22; border: 1px solid #30363D; border-left: 3px solid #6E56CF; border-radius: 6px; padding: 20px 24px; margin: 16px 0; }
.q-meta { font-size: 10px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #7D8590; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }
.q-meta .diff-easy { color: #3FB950; }
.q-meta .diff-medium { color: #D29922; }
.q-meta .diff-hard { color: #F85149; }
.q-text { font-size: 15px; line-height: 1.65; color: #E6EDF3; }

.example-label { font-size: 10px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #7D8590; margin-bottom: 8px; }

.score-card { background: #161B22; border: 1px solid #30363D; border-radius: 6px; padding: 20px 24px; margin: 12px 0; }
.score-line { font-family: 'JetBrains Mono', monospace; font-size: 22px; font-weight: 500; margin-bottom: 16px; letter-spacing: -0.02em; }
.score-high { color: #3FB950; }
.score-mid  { color: #D29922; }
.score-low  { color: #F85149; }
.score-section-label { font-size: 10px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #7D8590; margin: 14px 0 6px; }
.score-item { font-size: 13px; color: #E6EDF3; padding: 4px 0; display: flex; align-items: flex-start; gap: 8px; line-height: 1.5; }
.score-item .check { color: #3FB950; flex-shrink: 0; }
.score-item .cross { color: #F85149; flex-shrink: 0; }
.improvement-box { background: rgba(110, 86, 207, 0.08); border: 1px solid rgba(110, 86, 207, 0.25); border-radius: 4px; padding: 10px 14px; font-size: 13px; color: #A78BFA; margin-top: 12px; line-height: 1.55; }
.star-warning { background: rgba(210, 153, 34, 0.08); border: 1px solid rgba(210, 153, 34, 0.25); border-radius: 4px; padding: 10px 14px; font-size: 13px; color: #D29922; margin-top: 8px; }

.answer-bubble { background: #21262D; border: 1px solid #30363D; border-radius: 6px; padding: 14px 18px; font-size: 14px; color: #C9D1D9; margin: 8px 0; line-height: 1.6; }

.session-header { margin: 24px 0 20px; }
.session-avg { font-family: 'JetBrains Mono', monospace; font-size: 42px; font-weight: 500; letter-spacing: -0.03em; line-height: 1; }
.session-avg-label { font-size: 12px; color: #7D8590; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.06em; }
.topic-row { display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #21262D; font-size: 13px; }
.topic-row:last-child { border-bottom: none; }
.topic-name { color: #C9D1D9; }
.topic-score { font-family: 'JetBrains Mono', monospace; font-size: 13px; }

.plan-slot { display: flex; gap: 12px; padding: 8px 0; border-bottom: 1px solid #21262D; font-size: 13px; line-height: 1.5; }
.plan-slot:last-child { border-bottom: none; }
.plan-time { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #6E56CF; white-space: nowrap; padding-top: 1px; min-width: 140px; }
.plan-activity { color: #C9D1D9; }
.plan-task { font-size: 13px; color: #C9D1D9; padding: 5px 0; border-bottom: 1px solid #21262D; line-height: 1.5; }
.plan-task:last-child { border-bottom: none; }
.plan-task::before { content: "→ "; color: #6E56CF; }
.plan-resource { font-size: 12px; color: #7D8590; padding: 4px 0; }
.plan-resource::before { content: "↗ "; color: #3FB950; }
.done-when { background: rgba(63, 185, 80, 0.07); border: 1px solid rgba(63, 185, 80, 0.2); border-radius: 4px; padding: 9px 13px; font-size: 12px; color: #3FB950; margin-top: 10px; line-height: 1.5; }
.weak-banner { background: rgba(248, 81, 73, 0.07); border: 1px solid rgba(248, 81, 73, 0.2); border-radius: 4px; padding: 10px 14px; font-size: 13px; color: #F85149; margin: 12px 0; }
.pro-tip { background: rgba(110, 86, 207, 0.07); border: 1px solid rgba(110, 86, 207, 0.2); border-radius: 4px; padding: 10px 14px; font-size: 13px; color: #A78BFA; margin: 12px 0; line-height: 1.55; }

.setup-label { font-size: 11px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: #7D8590; margin-bottom: 6px; }

.stTextInput > div > div > input { background: #161B22 !important; border: 1px solid #30363D !important; border-radius: 5px !important; color: #E6EDF3 !important; font-family: 'Inter', sans-serif !important; font-size: 14px !important; padding: 10px 14px !important; }
.stTextInput > div > div > input:focus { border-color: #6E56CF !important; box-shadow: 0 0 0 3px rgba(110, 86, 207, 0.15) !important; outline: none !important; }
.stSelectbox > div > div { background: #161B22 !important; border: 1px solid #30363D !important; border-radius: 5px !important; color: #E6EDF3 !important; }
.stButton > button { background: #6E56CF !important; color: #fff !important; border: none !important; border-radius: 5px !important; font-size: 13px !important; font-weight: 500 !important; padding: 9px 20px !important; letter-spacing: 0.01em !important; transition: background 0.15s ease !important; }
.stButton > button:hover { background: #5B42BF !important; }
.stButton > button[kind="secondary"] { background: #21262D !important; color: #C9D1D9 !important; border: 1px solid #30363D !important; }
.stButton > button[kind="secondary"]:hover { background: #30363D !important; }
div[data-testid="stChatInput"] > div { background: #161B22 !important; border: 1px solid #30363D !important; border-radius: 6px !important; }
div[data-testid="stChatInput"] textarea { color: #E6EDF3 !important; font-family: 'Inter', sans-serif !important; font-size: 14px !important; }
div[data-testid="stExpander"] { background: #161B22 !important; border: 1px solid #30363D !important; border-radius: 6px !important; }
div[data-testid="stExpander"] summary { color: #E6EDF3 !important; font-size: 13px !important; font-weight: 500 !important; }
.stSuccess, .stInfo, .stWarning, .stError { border-radius: 4px !important; font-size: 13px !important; }
[data-testid="stSpinner"] { color: #6E56CF !important; }
</style>
""", unsafe_allow_html=True)


MAX_QUESTIONS = 5
VALID_ROLES   = ["SWE", "Data", "Product", "Analyst"]
VALID_ROUNDS  = {
    "SWE":     ["DSA", "System Design", "Behavioral"],
    "Data":    ["Technical", "Case", "Behavioral"],
    "Product": ["Case", "Behavioral"],
    "Analyst": ["Technical", "Case", "Behavioral"]
}


def is_injection(text: str) -> bool:
    try:
        flagged, _ = check_injection(text, use_llm=False)
        return flagged
    except Exception:
        return False


@st.cache_resource
def get_question_graph():
    try:
        return build_question_graph()
    except Exception as e:
        st.error(f"Failed to build question graph: {e}")
        return None


@st.cache_resource
def get_answer_graph():
    try:
        return build_answer_graph()
    except Exception as e:
        st.error(f"Failed to build answer graph: {e}")
        return None


def score_color_class(score: int) -> str:
    if score >= 7: return "score-high"
    if score >= 4: return "score-mid"
    return "score-low"


def score_symbol(score: int) -> str:
    if score >= 7: return "●"
    if score >= 4: return "◐"
    return "○"


def render_wordmark():
    st.markdown("""
    <div class="ps-wordmark">
        <div class="ps-diamond"></div>
        <div>
            <div class="ps-name">PrepSense</div>
            <div class="ps-tagline">AI interview coach</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_message(msg: dict):
    if msg["type"] == "question":
        difficulty = msg.get("caption", "").split("—")[-1].strip().lower() if "—" in msg.get("caption", "") else ""
        diff_class = f"diff-{difficulty}" if difficulty in ("easy", "medium", "hard") else ""
        caption    = msg.get("caption", "")
        q_num_part = caption.split("—")[0].strip() if "—" in caption else caption
        diff_span  = f'<span class="{diff_class}">{difficulty.upper()}</span>' if difficulty else ""

        st.markdown(f"""
        <div class="q-card">
            <div class="q-meta">{q_num_part} {diff_span}</div>
            <div class="q-text">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)

        if msg.get("examples"):
            st.markdown('<div class="example-label" style="margin-top:12px;">Examples</div>', unsafe_allow_html=True)
            for ex in msg["examples"]:
                inp = ex.get("input", "")
                out = ex.get("output", "")
                if isinstance(inp, (dict, list)):
                    import json as _j
                    inp = _j.dumps(inp, indent=2)
                if isinstance(out, (dict, list)):
                    import json as _j
                    out = _j.dumps(out, indent=2)
                st.code(f"Input:  {inp}\nOutput: {out}", language="text")

    elif msg["type"] == "answer":
        st.markdown(f'<div class="answer-bubble">{msg["content"]}</div>', unsafe_allow_html=True)

    elif msg["type"] == "evaluation":
        score     = msg["score"]
        cls       = score_color_class(score)
        sym       = score_symbol(score)
        right     = msg.get("what_was_right", [])
        missing   = msg.get("what_was_missing", [])
        tip       = msg.get("improvement_tip", "")
        star_warn = msg.get("star_warning", False)

        st.markdown(f'<div class="score-card"><div class="score-line {cls}">{sym} {score} / 10</div>', unsafe_allow_html=True)

        if right:
            st.markdown('<div class="score-section-label">What you got right</div>', unsafe_allow_html=True)
            for p in right:
                st.markdown(f'<div class="score-item"><span class="check">✓</span> {p}</div>', unsafe_allow_html=True)

        if missing:
            st.markdown('<div class="score-section-label">What was missing</div>', unsafe_allow_html=True)
            for p in missing:
                st.markdown(f'<div class="score-item"><span class="cross">✗</span> {p}</div>', unsafe_allow_html=True)

        if tip:
            st.markdown(f'<div class="improvement-box">💬 {tip}</div>', unsafe_allow_html=True)

        if star_warn:
            st.markdown('<div class="star-warning">⚠ Answer was missing some STAR components.</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


def render_heatmap(user_id: str, company: str, role: str):
    try:
        import plotly.graph_objects as go

        history = get_session_history(user_id, company, role)
        if not history:
            st.markdown('<p style="color:#7D8590;font-size:13px;">Complete more sessions to see your heatmap.</p>', unsafe_allow_html=True)
            return

        all_topics = sorted(set(
            s.get("topic", "unknown").lower()
            for sess in history for s in sess["scores"]
        ))
        session_labels = [f"S{i+1}" for i in range(len(history))]

        matrix = []
        for topic in all_topics:
            row = []
            for sess in history:
                scores = [s["score"] for s in sess["scores"] if s.get("topic", "").lower() == topic]
                row.append(round(sum(scores)/len(scores), 1) if scores else None)
            matrix.append(row)

        z = [[v if v is not None else -1 for v in row] for row in matrix]

        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=session_labels,
            y=all_topics,
            colorscale=[
                [0.0, "#1a1a1a"], [0.1, "#3a0d0d"], [0.35, "#3a2a0a"],
                [0.6, "#2d3a0d"], [0.8, "#0d4a1f"], [1.0, "#0d6b2a"],
            ],
            zmin=-1, zmax=10,
            text=[[str(v) if v != -1 else "—" for v in row] for row in z],
            texttemplate="%{text}",
            textfont={"family": "JetBrains Mono", "size": 12, "color": "#E6EDF3"},
            showscale=False,
            hoverongaps=False,
            hovertemplate="Topic: %{y}<br>Session: %{x}<br>Score: %{text}<extra></extra>"
        ))

        fig.update_layout(
            plot_bgcolor="#0D1117", paper_bgcolor="#0D1117",
            font=dict(family="Inter", color="#7D8590", size=11),
            margin=dict(l=0, r=0, t=8, b=0),
            height=max(120, len(all_topics) * 38 + 40),
            xaxis=dict(side="top", tickfont=dict(family="JetBrains Mono", size=11, color="#7D8590"), gridcolor="#21262D", linecolor="#21262D"),
            yaxis=dict(tickfont=dict(family="Inter", size=11, color="#C9D1D9"), gridcolor="#21262D", linecolor="#21262D")
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    except ImportError:
        st.info("Install plotly for the heatmap: `pip install plotly`")
    except Exception as e:
        st.markdown(f'<p style="color:#F85149;font-size:12px;">Heatmap error: {e}</p>', unsafe_allow_html=True)


def render_study_plan(plan: dict):
    company = plan.get("target_company", "")
    role    = plan.get("target_role", "")
    hours   = plan.get("total_hours", 35)

    st.markdown(f"""
    <div style="margin: 24px 0 16px;">
        <div style="font-size:16px;font-weight:600;color:#E6EDF3;letter-spacing:-0.02em;">7-Day Plan — {company} · {role}</div>
        <div style="font-size:11px;color:#7D8590;margin-top:3px;font-family:'JetBrains Mono',monospace;">{hours}h total · ~5h/day</div>
    </div>
    """, unsafe_allow_html=True)

    if plan.get("weak_areas"):
        st.markdown(f'<div class="weak-banner">Targeting weak areas: {", ".join(plan["weak_areas"])}</div>', unsafe_allow_html=True)
    if plan.get("final_tip"):
        st.markdown(f'<div class="pro-tip">◆ {plan["final_tip"]}</div>', unsafe_allow_html=True)

    days = plan.get("days", [])
    if not days:
        st.markdown('<p style="color:#7D8590;font-size:13px;">Complete more sessions to generate a personalised plan.</p>', unsafe_allow_html=True)
        return

    for day in days:
        day_num   = day.get("day", "")
        focus     = day.get("focus", "")
        day_hours = day.get("hours", 5)
        schedule  = day.get("schedule", [])
        tasks     = day.get("tasks", [])
        resources = day.get("resources", [])
        success   = day.get("success_metric", "")

        with st.expander(f"Day {day_num} — {focus}  ·  {day_hours}h", expanded=(day_num == 1)):
            if schedule:
                st.markdown('<div class="score-section-label">Schedule</div>', unsafe_allow_html=True)
                st.markdown("".join([
                    f'<div class="plan-slot"><span class="plan-time">{s.get("hour","")}</span><span class="plan-activity">{s.get("activity","")}</span></div>'
                    for s in schedule
                ]), unsafe_allow_html=True)
            if tasks:
                st.markdown('<div class="score-section-label" style="margin-top:14px;">Tasks</div>', unsafe_allow_html=True)
                st.markdown("".join([f'<div class="plan-task">{t}</div>' for t in tasks]), unsafe_allow_html=True)
            if resources:
                st.markdown('<div class="score-section-label" style="margin-top:14px;">Resources</div>', unsafe_allow_html=True)
                st.markdown("".join([f'<div class="plan-resource">{r}</div>' for r in resources]), unsafe_allow_html=True)
            if success:
                st.markdown(f'<div class="done-when">Done when: {success}</div>', unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
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


render_wordmark()

# ── Setup screen ───────────────────────────────────────────────────────────────
if not st.session_state.setup_done:
    st.markdown('<div class="ps-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#7D8590;font-size:13px;margin-bottom:20px;">Configure your session below. PrepSense adapts difficulty and feedback to your performance.</p>', unsafe_allow_html=True)

    st.markdown('<div class="setup-label">Company</div>', unsafe_allow_html=True)
    company = st.text_input("Company", placeholder="Amazon, Google, Microsoft...", label_visibility="collapsed")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="setup-label">Role</div>', unsafe_allow_html=True)
        role = st.selectbox("Role", VALID_ROLES, label_visibility="collapsed")
    with col2:
        st.markdown('<div class="setup-label">Round</div>', unsafe_allow_html=True)
        round_type = st.selectbox("Round", VALID_ROUNDS[role], label_visibility="collapsed")

    st.markdown('<div class="setup-label" style="margin-top:12px;">Topic</div>', unsafe_allow_html=True)
    topic = st.text_input("Topic", placeholder="e.g. dynamic programming, SQL joins, product metrics", label_visibility="collapsed")

    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

    col_jd, col_start = st.columns([2, 1])
    with col_jd:
        if st.button("Load job description", type="secondary", use_container_width=True):
            if not company:
                st.error("Enter a company name first.")
            else:
                PRELOADED = ["Amazon", "Google", "Microsoft", "Adobe"]
                if company in PRELOADED:
                    st.session_state.jd_preview = {"found": True, "source": "preloaded", "skills": [], "titles": [f"{company} Software Engineer"]}
                else:
                    with st.spinner(f"Fetching {company} JD..."):
                        try:
                            from rag.jdfetcher import fetch_and_store_jd
                            st.session_state.jd_preview = fetch_and_store_jd(company)
                        except Exception as e:
                            st.session_state.jd_preview = {"found": False, "error": str(e)}

    with col_start:
        start_clicked = st.button("Start session →", type="primary", use_container_width=True)

    if st.session_state.jd_preview:
        preview = st.session_state.jd_preview
        if preview.get("source") == "preloaded":
            st.success(f"✓ {company} — using curated question bank")
        elif preview.get("found"):
            st.success(f"✓ Live JD found for {company}")
            if preview.get("skills"):
                st.markdown('<div class="setup-label" style="margin-top:12px;">Skills from JD</div>', unsafe_allow_html=True)
                cols = st.columns(3)
                for i, skill in enumerate(preview["skills"][:12]):
                    with cols[i % 3]:
                        st.markdown(f'<span style="font-size:12px;color:#7D8590;">· {skill}</span>', unsafe_allow_html=True)
        else:
            st.warning(f"No JD found for {company} — will use generic question bank")

    if start_clicked:
        if not company or not topic:
            st.error("Company and topic are required.")
        elif is_injection(company) or is_injection(topic):
            st.error("That input looks suspicious. Please enter a valid company or topic.")
        else:
            st.session_state.graph_state        = create_initial_state(company=company, role=role, round_type=round_type, topic=topic, user_id="default_user")
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

    badges = "".join([
        f'<span class="badge active">{company}</span>',
        f'<span class="badge">{role}</span>',
        f'<span class="badge">{round_type}</span>',
        f'<span class="badge">{topic}</span>',
    ])
    st.markdown(f'<div class="badge-row">{badges}</div>', unsafe_allow_html=True)

    if not st.session_state.session_complete:
        pct = int((q_num / max_q) * 100) if max_q > 0 else 0
        st.markdown(f"""
        <div class="progress-wrap">
            <div class="progress-label">Progress — {q_num} of {max_q}</div>
            <div class="progress-track"><div class="progress-fill" style="width:{pct}%"></div></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="ps-divider"></div>', unsafe_allow_html=True)

    for msg in st.session_state.chat_messages:
        render_message(msg)

    # ── Session complete ───────────────────────────────────────────────────────
    if st.session_state.session_complete:
        scores = state["session_scores"]
        avg    = round(sum(s["score"] for s in scores) / len(scores), 1) if scores else 0

        try:
            save_session(user_id=user_id, company=company, role=role, round_type=round_type, scores=scores)
        except Exception as e:
            st.warning(f"Could not save session: {e}")

        try:
            weak_areas = get_weak_areas(user_id, company, role)
            avg_scores = get_avg_score_by_topic(user_id, company, role)
        except Exception:
            weak_areas, avg_scores = [], {}

        cls = score_color_class(int(avg))
        st.markdown(f"""
        <div class="session-header">
            <div class="session-avg {cls}">{avg}</div>
            <div class="session-avg-label">Average score this session</div>
        </div>
        """, unsafe_allow_html=True)

        if weak_areas:
            st.markdown(f'<div class="weak-banner">Weak areas: {", ".join(weak_areas)}</div>', unsafe_allow_html=True)

        st.markdown('<div class="score-section-label">This session</div>', unsafe_allow_html=True)
        rows = ""
        for s in scores:
            sc   = s["score"]
            cls2 = score_color_class(sc)
            rows += f'<div class="topic-row"><span class="topic-name">{s["topic"]}</span><span class="topic-score {cls2}">{sc} / 10</span></div>'
        st.markdown(rows, unsafe_allow_html=True)

        if avg_scores:
            st.markdown('<div class="score-section-label" style="margin-top:20px;">All-time performance heatmap</div>', unsafe_allow_html=True)
            render_heatmap(user_id, company, role)

        st.markdown('<div class="ps-divider"></div>', unsafe_allow_html=True)

        if not st.session_state.show_study_plan:
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Generate study plan", type="primary", use_container_width=True):
                    with st.spinner("Building your study plan..."):
                        try:
                            # Pass session_scores so study plan is grounded in THIS session
                            st.session_state.study_plan      = generate_study_plan(user_id, company, role, session_scores=scores)
                            st.session_state.show_study_plan = True
                        except Exception as e:
                            st.error(f"Could not generate study plan: {e}")
                    st.rerun()
            with c2:
                if st.button("New session", type="secondary", use_container_width=True):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()
        else:
            render_study_plan(st.session_state.study_plan)
            st.markdown('<div class="ps-divider"></div>', unsafe_allow_html=True)
            if st.button("New session", type="secondary"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

    # ── Generate next question ─────────────────────────────────────────────────
    elif not st.session_state.waiting_for_answer:
        q_graph = get_question_graph()
        if q_graph is None:
            st.error("Interview graph failed to load. Please refresh.")
            st.stop()

        with st.spinner(f"Preparing question {q_num + 1} of {max_q}..."):
            try:
                state = q_graph.invoke(state)
                st.session_state.graph_state = state
            except Exception as e:
                st.error(f"Failed to generate question: {e}")
                st.markdown(f'<pre style="color:#F85149;font-size:11px;">{traceback.format_exc()}</pre>', unsafe_allow_html=True)
                st.stop()

        question_text = state["current_question"]
        q_num         = state["question_number"]
        difficulty    = state["current_difficulty"]
        examples      = state.get("current_examples", [])

        st.session_state.chat_messages.append({
            "role":     "assistant",
            "type":     "question",
            "caption":  f"Question {q_num} of {max_q} — {difficulty}",
            "content":  question_text,
            "examples": examples
        })
        st.session_state.waiting_for_answer = True
        st.rerun()

    # ── Wait for answer ────────────────────────────────────────────────────────
    else:
        answer = st.chat_input("Type your answer...")
        if answer:
            if is_injection(answer):
                st.warning("That input looks like a prompt injection attempt. Please answer the interview question.")
                st.stop()

            st.session_state.chat_messages.append({"role": "user", "type": "answer", "content": answer})
            state["user_answer"] = answer

            a_graph = get_answer_graph()
            if a_graph is None:
                st.error("Answer graph failed to load. Please refresh.")
                st.stop()

            with st.spinner("Evaluating..."):
                try:
                    state = a_graph.invoke(state)
                    st.session_state.graph_state = state
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")
                    st.stop()

            if state.get("clarification_response"):
                st.session_state.chat_messages.append({
                    "role":     "assistant",
                    "type":     "question",
                    "caption":  "Clarification",
                    "content":  state["clarification_response"],
                    "examples": []
                })
                st.session_state.graph_state["clarification_response"] = ""
                st.session_state.waiting_for_answer = True
                st.rerun()

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
                    "star_warning":     round_type == "Behavioral" and not eval_result.get("star_complete", True),
                    "content":          f"Score: {score}/10"
                })

                if state["question_number"] >= state["max_questions"]:
                    st.session_state.session_complete = True
                else:
                    st.session_state.waiting_for_answer = False

                st.rerun()