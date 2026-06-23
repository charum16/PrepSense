# agents/study_plan_agent.py
from groq import Groq
from dotenv import load_dotenv
import os
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memory.session_store import get_weak_areas, get_avg_score_by_topic
from rag.retriever_v1 import retrieve_questions

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

COMPANY_TOPICS = {
    "Amazon": {
        "SWE": [
            "Arrays & Two Pointers",
            "Hash Maps & Sets",
            "Trees & BST",
            "Dynamic Programming",
            "Graphs & BFS/DFS",
            "System Design at Amazon scale",
            "Leadership Principles behavioral"
        ],
        "Data": ["SQL window functions", "A/B testing & metrics", "Data pipeline design", "Business case analysis", "Statistics & probability"],
        "Product": ["Customer obsession framework", "Metrics & success criteria", "Product tradeoffs at scale", "Launch strategy", "Behavioral STAR"],
        "Analyst": ["SQL aggregations & joins", "Funnel & cohort analysis", "Business metrics definition", "Data visualization", "Stakeholder communication"]
    },
    "Google": {
        "SWE": [
            "Arrays & Strings",
            "Graphs & BFS/DFS",
            "Dynamic Programming",
            "Binary Search & Divide/Conquer",
            "System Design at Google scale",
            "Concurrency & distributed systems",
            "Code quality & testing"
        ],
        "Data": ["SQL advanced queries", "Statistical inference", "Experiment design", "Python/R analysis", "Data modeling"],
        "Product": ["Product strategy & vision", "Metrics & success criteria", "Technical depth", "User empathy & research", "Cross-functional execution"],
        "Analyst": ["SQL & data extraction", "Statistical analysis", "Visualization & storytelling", "Business insight", "Stakeholder communication"]
    },
    "Microsoft": {
        "SWE": [
            "Arrays & Linked Lists",
            "Trees & Recursion",
            "Dynamic Programming",
            "System Design",
            "OOP & design patterns",
            "Cloud & Azure basics",
            "Growth mindset behavioral"
        ],
        "Data": ["SQL & T-SQL", "Power BI", "Statistical modeling", "Azure data services", "Business storytelling"],
        "Product": ["Empathy & design thinking", "Feature prioritization", "Metrics & KPIs", "Competitive analysis", "Behavioral"],
        "Analyst": ["SQL", "Excel advanced", "Data visualization", "Business analysis", "Presentation skills"]
    },
    "Adobe": {
        "SWE": ["Arrays & Strings", "Trees & Graphs", "Dynamic Programming", "OOP design", "Performance optimization", "Behavioral"],
        "Data": ["SQL", "Statistical analysis", "Experimentation", "Marketing analytics", "Visualization"],
        "Product": ["Creative tools knowledge", "User research & empathy", "Metrics & OKRs", "Roadmap planning", "Behavioral"],
        "Analyst": ["SQL", "Adobe Analytics", "Business metrics", "Reporting & dashboards", "Stakeholder management"]
    },
    "Meta": {
        "SWE": [
            "Arrays & Hash Maps",
            "Trees & Graphs",
            "Dynamic Programming",
            "System Design at Meta scale",
            "Behavioral & culture fit",
            "Distributed systems",
            "Coding speed & accuracy"
        ],
        "Data": ["SQL at scale", "Experimentation & A/B testing", "Product metrics", "Statistical modeling", "Python analysis"],
        "Product": ["Product sense", "Metrics & north star", "Execution & prioritization", "Leadership & influence", "Behavioral"],
        "Analyst": ["SQL", "Business metrics", "Experimentation", "Visualization", "Communication"]
    }
}

DEFAULT_TOPICS = {
    "SWE": ["Arrays & Hash Maps", "Trees & Graphs", "Dynamic Programming", "System Design", "Behavioral"],
    "Data": ["SQL", "Statistics", "Experimentation", "Data modeling", "Communication"],
    "Product": ["Product sense", "Metrics", "Strategy", "Execution", "Behavioral"],
    "Analyst": ["SQL", "Business analysis", "Visualization", "Statistics", "Communication"]
}

COMPANY_RESOURCES = {
    "Amazon": {
        "SWE": [
            "NeetCode 150 — neetcode.io (free, structured by pattern)",
            "LeetCode Amazon tag — leetcode.com/company/amazon",
            "Amazon Leadership Principles — amazon.jobs/principles",
            "AlgoExpert behavioral section — algoexpert.io",
            "Grokking System Design — educative.io/courses/grokking-the-system-design-interview"
        ]
    },
    "Google": {
        "SWE": [
            "NeetCode 150 — neetcode.io",
            "LeetCode Google tag — leetcode.com/company/google",
            "Tech Interview Handbook — techinterviewhandbook.org",
            "Google SWE book — 'Cracking the Coding Interview' by Gayle McDowell",
            "Grokking System Design — educative.io"
        ]
    }
}

def _get_resources(company: str, role: str) -> list:
    return COMPANY_RESOURCES.get(company, {}).get(role, [
        "NeetCode 150 — neetcode.io (free, structured by pattern)",
        f"LeetCode {company} tag — leetcode.com/company/{company.lower()}",
        "Tech Interview Handbook — techinterviewhandbook.org",
        "Grokking System Design — educative.io/courses/grokking-the-system-design-interview",
        "Pramp mock interviews — pramp.com (free peer mock interviews)"
    ])


def generate_study_plan(user_id: str, company: str, role: str, session_scores: list = None) -> dict:
    try:
        weak_areas = get_weak_areas(user_id, company, role)
        avg_scores = get_avg_score_by_topic(user_id, company, role)
    except TypeError:
        weak_areas = get_weak_areas(user_id)
        avg_scores = get_avg_score_by_topic(user_id)

    company_topics = COMPANY_TOPICS.get(company, {}).get(role, DEFAULT_TOPICS.get(role, []))
    resources      = _get_resources(company, role)

    # If no stored weak areas, derive them from the current session scores directly
    if not weak_areas and session_scores:
        topic_avgs = {}
        for s in session_scores:
            t = s.get("topic", "").strip().lower()
            if t:
                topic_avgs.setdefault(t, []).append(s["score"])
        weak_areas = [
            t for t, scores in topic_avgs.items()
            if sum(scores) / len(scores) < 6
        ]
        # Also merge into avg_scores so the plan prompt has data
        for t, scores in topic_avgs.items():
            avg_scores[t] = round(sum(scores) / len(scores), 1)

    if not weak_areas:
        weak_areas = [company_topics[0]] if company_topics else ["arrays & hash maps"]

    scores_str = "\n".join([f"- {t}: {s}/10" for t, s in avg_scores.items()]) if avg_scores else "No history yet — assume beginner level."

    # Fetch real practice questions for weak areas
    practice_qs = {}
    for topic in weak_areas[:3]:
        try:
            results = retrieve_questions(company, role, "DSA", topic, n_results=3)
            if not results:
                results = retrieve_questions("Generic", role, "DSA", topic, n_results=3)
            practice_qs[topic] = [r["question"] for r in results if r.get("question")]
        except Exception:
            practice_qs[topic] = []

    practice_str = ""
    for topic, qs in practice_qs.items():
        if qs:
            practice_str += f"\n{topic}:\n" + "\n".join([f"  - {q}" for q in qs])

    remaining_topics = [t for t in company_topics if t not in weak_areas]

    prompt = f"""You are an expert technical interview coach who has helped hundreds of candidates get offers at {company}.

── CANDIDATE PROFILE ─────────────────────────────────────────────────────────
Target company: {company}
Target role: {role}
Weak areas (scored below 6/10 in mock sessions): {', '.join(weak_areas)}
Score history:
{scores_str}

── REAL PRACTICE QUESTIONS FROM {company.upper()} QUESTION BANK ──────────────
{practice_str if practice_str else 'Use standard interview prep.'}

── {company.upper()} IMPORTANT TOPICS FOR {role} ─────────────────────────────
{', '.join(company_topics)}

── AVAILABLE RESOURCES ───────────────────────────────────────────────────────
{chr(10).join(resources)}

── YOUR TASK ─────────────────────────────────────────────────────────────────
Generate a highly specific, actionable 7-day study plan with 4-5 hours of daily dedication.

STRUCTURE REQUIREMENTS:
- Days 1-2: EXCLUSIVELY target weak areas: {', '.join(weak_areas[:2])}
- Days 3-5: Cover {company}-specific important topics: {', '.join(remaining_topics[:3]) if remaining_topics else ', '.join(company_topics[2:5])}
- Day 6: Full mock interview simulation day
- Day 7: Weak area review + confidence building + final prep

QUALITY BAR — each day must have:
1. A clear focus with a specific sub-goal (not "study trees" but "master tree traversals: inorder, preorder, postorder, level-order BFS")
2. A hour-by-hour schedule (Hour 1: ..., Hour 2: ..., Hour 3: ..., Hour 4-5: ...)
3. Exactly 3-5 specific tasks with concrete deliverables (specific LeetCode problem numbers, specific SQL queries to write, specific concepts to master)
4. 2-3 specific resources with URLs
5. A success metric — how do you know you've completed the day successfully?

BANNED VAGUE TASKS (never use these):
- "Solve 3 problems on LeetCode" → instead: "Solve LeetCode #1 Two Sum, #49 Group Anagrams, #347 Top K Frequent Elements — all using hash maps, aim for O(n) solution"
- "Review Big-O notation" → instead: "Write time/space complexity for 10 solutions you've already coded"  
- "Practice whiteboarding" → instead: "Solve LeetCode #200 Number of Islands on paper without IDE, then verify"
- "Study data structures" → instead: "Implement a HashMap from scratch with get/set/delete, then solve #706 Design HashMap"

CONTEXT: Frame tasks in {company}'s world where possible.
Example for Amazon: "Solve LeetCode #146 LRU Cache — this is exactly how Amazon's ElastiCache eviction works"
Example for Google: "Solve LeetCode #295 Find Median from Data Stream — used in Google's real-time analytics pipelines"

Respond in this exact JSON only. No text before or after:
{{
    "target_company": "{company}",
    "target_role": "{role}",
    "weak_areas": {json.dumps(weak_areas)},
    "total_hours": 35,
    "days": [
        {{
            "day": 1,
            "focus": "specific topic — specific sub-goal",
            "hours": 5,
            "schedule": [
                {{"hour": "Hour 1 (9:00-10:00)", "activity": "specific activity"}},
                {{"hour": "Hour 2 (10:00-11:00)", "activity": "specific activity"}},
                {{"hour": "Hour 3 (11:00-12:00)", "activity": "specific activity"}},
                {{"hour": "Hour 4-5 (1:00-3:00)", "activity": "specific activity"}}
            ],
            "tasks": [
                "specific task with problem numbers or exact concepts",
                "specific task with problem numbers or exact concepts",
                "specific task with problem numbers or exact concepts"
            ],
            "resources": [
                "Resource name — URL",
                "Resource name — URL"
            ],
            "success_metric": "By end of day you should be able to: ..."
        }}
    ],
    "final_tip": "one specific, tactical tip for {company} {role} interviews that most candidates miss"
}}

Generate exactly 7 days. Be specific. Be concrete. No filler."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a technical interview coach. Respond in valid JSON only. No markdown, no text before or after the JSON."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.4,
        max_tokens=4000,
        stream=False
    )

    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except Exception:
        cleaned = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return json.loads(cleaned)
        except Exception:
            return {
                "target_company": company,
                "target_role":    role,
                "weak_areas":     weak_areas,
                "total_hours":    35,
                "days":           [],
                "final_tip":      f"Focus on {', '.join(weak_areas)} before your {company} interview."
            }


if __name__ == "__main__":
    plan = generate_study_plan("default_user", "Amazon", "SWE")
    print(json.dumps(plan, indent=2))