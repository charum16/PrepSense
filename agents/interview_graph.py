# agents/interview_graph.py
from langsmith import traceable
import os
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.retriever_v1 import retrieve_questions
from api.evaluator import evaluate_answer
from api.cache import get_cached_question, cache_question
from api.guardrails import check_injection


class InterviewState(TypedDict):
    company: str
    role: str
    round_type: str
    topic: str
    user_id: str
    messages: Annotated[list, add_messages]
    current_question: str
    current_difficulty: str
    current_examples: list
    rag_context: list
    current_score: int
    session_scores: list
    question_number: int
    max_questions: int
    user_answer: str
    last_evaluation: dict
    clarification_response: str
    asked_questions: list


COMPANY_BAR = {
    "Amazon": {
        "persona": "Amazon interviewer who values optimal solutions and Leadership Principles. Amazon's bar is high even for Easy — they expect candidates to immediately think of hash maps, two pointers, or sliding window, never brute force.",
        "easy_bar": "Two Sum, Valid Parentheses, Merge Sorted Arrays, Contains Duplicate — requires knowing the RIGHT data structure immediately",
        "medium_bar": "LRU Cache, Number of Islands, Product of Array Except Self, Longest Substring Without Repeating Characters",
        "hard_bar": "Median of Data Streams, Serialize/Deserialize Binary Tree, Word Ladder, Trapping Rain Water",
        "context": "Amazon's delivery routing, product catalog, order processing, recommendation engine, Prime membership systems, AWS infrastructure"
    },
    "Google": {
        "persona": "Google interviewer who values elegant, scalable code and clear communication. Google expects candidates to think about edge cases and explain their reasoning out loud.",
        "easy_bar": "Reverse Linked List, Binary Search, Valid Palindrome, First Bad Version — clean code and complexity analysis required",
        "medium_bar": "Search in Rotated Array, Find Peak Element, Decode Ways, Combination Sum",
        "hard_bar": "Skyline Problem, Russian Doll Envelopes, Word Search II, Alien Dictionary",
        "context": "Google's search indexing, Maps routing, YouTube recommendations, Gmail spam detection, autocomplete systems, PageRank"
    },
    "Microsoft": {
        "persona": "Microsoft interviewer who values problem-solving process and growth mindset. Microsoft wants to see how you think and iterate, not just the final answer.",
        "easy_bar": "Reverse String, Fibonacci with memoization, Valid Anagram, Move Zeroes — expects discussion of tradeoffs",
        "medium_bar": "Clone Graph, Course Schedule, Unique Paths, Jump Game",
        "hard_bar": "Edit Distance, Regular Expression Matching, Burst Balloons, Minimum Window Substring",
        "context": "Microsoft's Azure cloud services, Office 365 document processing, Teams chat infrastructure, Xbox gaming systems, Bing search"
    },
    "Adobe": {
        "persona": "Adobe interviewer who values creative problem-solving and attention to detail. Adobe expects candidates to think about performance for large media files.",
        "easy_bar": "Two Sum, Maximum Subarray, Best Time to Buy/Sell Stock, Climbing Stairs",
        "medium_bar": "Spiral Matrix, Rotate Image, Merge Intervals, Find All Anagrams",
        "hard_bar": "Largest Rectangle in Histogram, Maximal Rectangle, Edit Distance, Text Justification",
        "context": "Adobe's image processing pipelines, PDF rendering, video editing timelines, Creative Cloud file sync, font rendering systems, Photoshop layers"
    },
    "Meta": {
        "persona": "Meta interviewer who values speed and scalability. Meta's bar emphasizes graph problems, trees, and systems that handle billions of users.",
        "easy_bar": "Merge Two Sorted Lists, Symmetric Tree, Path Sum, Invert Binary Tree — expects optimal from the start",
        "medium_bar": "Binary Tree Level Order Traversal, Accounts Merge, Random Pick with Weight, Subarray Sum Equals K",
        "hard_bar": "Sliding Window Maximum, Minimum Cost to Connect All Points, Count of Smaller Numbers After Self",
        "context": "Meta's News Feed ranking, Instagram photo storage, WhatsApp message delivery, ad targeting systems, social graph traversal"
    },
    "Netflix": {
        "persona": "Netflix interviewer focused on distributed systems and personalization at scale.",
        "easy_bar": "Valid Parentheses, Linked List Cycle, Missing Number, Single Number",
        "medium_bar": "Top K Frequent Elements, Design Hit Counter, Find Duplicate Number, Longest Consecutive Sequence",
        "hard_bar": "LFU Cache, Design Search Autocomplete System, Stream of Characters",
        "context": "Netflix's video streaming pipelines, recommendation algorithms, A/B testing infrastructure, content delivery networks, watch history systems"
    }
}

DEFAULT_BAR = {
    "persona": "a senior technical interviewer with high standards. Expect candidates to know optimal solutions, not brute force.",
    "easy_bar": "problems requiring basic data structures used correctly — hash maps, two pointers, sliding window",
    "medium_bar": "problems requiring combining 2+ concepts — BFS/DFS + memoization, binary search + greedy",
    "hard_bar": "problems requiring advanced techniques — segment trees, tries, complex DP, topological sort",
    "context": "real engineering challenges this company would face at scale"
}


def _get_difficulty(state: InterviewState) -> str:
    scores = state.get("session_scores", [])
    if not scores:
        return "easy"
    last_score = scores[-1]["score"]
    if last_score >= 8:
        return "hard"
    elif last_score >= 5:
        return "medium"
    else:
        return "easy"


def _to_groq_messages(raw_messages: list) -> list:
    result = []
    for m in raw_messages:
        if isinstance(m, dict):
            if "role" in m and "content" in m:
                result.append({"role": m["role"], "content": m["content"]})
        else:
            role = "assistant" if m.__class__.__name__ == "AIMessage" else "user"
            result.append({"role": role, "content": m.content})
    return result


@traceable(name="retrieve_context")
def retrieve_context_node(state: InterviewState) -> dict:
    difficulty = _get_difficulty(state)
    context = retrieve_questions(
        company=state["company"],
        role=state["role"],
        round_type=state["round_type"],
        topic=state["topic"],
        n_results=5
    )
    return {
        "rag_context": context,
        "current_difficulty": difficulty
    }


@traceable(name="generate_question")
def generate_question_node(state: InterviewState) -> dict:
    from groq import Groq
    from dotenv import load_dotenv
    import json
    import random
    load_dotenv()

    client     = Groq(api_key=os.getenv("GROQ_API_KEY"))
    company    = state["company"]
    role       = state["role"]
    round_type = state["round_type"]
    topic      = state["topic"]
    difficulty = state["current_difficulty"]
    rag        = state["rag_context"]
    asked      = state.get("asked_questions", [])
    bar        = COMPANY_BAR.get(company, DEFAULT_BAR)

    asked_str = "\n".join([f"- {q}" for q in asked]) if asked else "None yet."

    # Cache check
    cached = get_cached_question(company, role, round_type, topic, difficulty)
    if cached and cached.get("question") not in asked:
        print(f"[cache] Returning cached question for {company} {round_type} {topic}")
        question_text = cached["question"]
        examples      = cached.get("examples", [])
        return {
            "current_question": question_text,
            "current_examples": examples,
            "question_number":  state["question_number"] + 1,
            "asked_questions":  asked + [question_text],
            "messages":         [{"role": "assistant", "content": question_text}]
        }

    unused_rag = [r for r in rag if r["question"] not in asked]
    seed_question = random.choice(unused_rag)["question"] if unused_rag else (random.choice(rag)["question"] if rag else None)

    # CRITICAL: company name isolation enforced in both system and user message
    system = (
        f"You are a senior {role} interviewer at {company}. "
        f"You ONLY ask questions relevant to {company}. "
        f"You NEVER mention any other company name — not Google, Amazon, Microsoft, Meta, Adobe, or any other. "
        f"You frame all scenarios using {company}'s own products and systems: {bar['context']}. "
        f"Respond in JSON only. No text before or after the JSON."
    )

    if seed_question:
        user_msg = f"""Here is a real {company} interview question from our verified database:

"{seed_question}"

Create a NEW question that:
1. Tests the same concept about {topic} but with a different scenario or constraint
2. Is at {difficulty} difficulty — at {company}, {difficulty} questions look like: {bar[difficulty + '_bar']}
3. Frames the problem ONLY in {company}'s context using {company}'s own products/systems — NEVER reference any other company
4. Is NOT similar to any of these already-asked questions:
{asked_str}

For DSA questions include exactly 2 concrete input/output examples with real values.

Respond in this exact JSON only:
{{
    "question": "your new question here — framed ONLY in {company} context",
    "difficulty": "{difficulty}",
    "topic": "{topic}",
    "examples": [
        {{"input": "...", "output": "..."}},
        {{"input": "...", "output": "..."}}
    ]
}}"""
    else:
        user_msg = (
            f"Ask me a {difficulty} {round_type} question about {topic} for a {company} {role} interview. "
            f"Frame it ONLY in {company}'s context. Never mention any other company."
        )

    history = _to_groq_messages(state.get("messages", []))

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system}] + history + [
            {"role": "user", "content": user_msg}
        ],
        temperature=0.7,
        max_tokens=700,
        stream=False
    )

    raw = response.choices[0].message.content
    try:
        result = json.loads(raw)
    except Exception:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            result = json.loads(cleaned)
        except Exception:
            result = {"question": raw, "difficulty": difficulty, "topic": topic, "examples": []}

    question_text = result.get("question", raw)
    examples      = result.get("examples", [])

    # Handle nested JSON in question field
    if isinstance(question_text, str):
        stripped = question_text.strip()
        if stripped.startswith("{") or stripped.startswith("```"):
            try:
                cleaned_inner = stripped.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                inner = json.loads(cleaned_inner)
                if isinstance(inner, dict) and "question" in inner:
                    question_text = inner["question"]
                    if not examples:
                        examples = inner.get("examples", [])
            except Exception:
                pass

    # DSA safety net
    if round_type == "DSA" and (not examples or len(examples) < 2):
        retry = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a coding interviewer. Respond in JSON only."},
                {"role": "user", "content": f"""Add exactly 2 concrete input/output examples to this DSA question and return the full JSON.

Question: {question_text}

Return:
{{
    "question": "{question_text}",
    "difficulty": "{difficulty}",
    "topic": "{topic}",
    "examples": [
        {{"input": "example input 1", "output": "example output 1"}},
        {{"input": "example input 2", "output": "example output 2"}}
    ]
}}"""}
            ],
            temperature=0.3,
            max_tokens=400,
            stream=False
        )
        try:
            retry_result = json.loads(
                retry.choices[0].message.content.strip()
                    .removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            )
            examples = retry_result.get("examples", examples)
        except Exception:
            pass

    cache_question(company, role, round_type, topic, difficulty, {
        "question": question_text,
        "examples": examples
    })

    return {
        "current_question": question_text,
        "current_examples": examples,
        "question_number":  state["question_number"] + 1,
        "asked_questions":  asked + [question_text],
        "messages":         [{"role": "assistant", "content": question_text}]
    }


def route_answer_node(state: InterviewState) -> dict:
    return {}


def is_clarification_or_answer(state: InterviewState) -> str:
    text = state.get("user_answer", "").strip().lower()
    question_starters = (
        "what", "how", "can i", "could i", "should i",
        "do you", "is it", "are we", "can you", "clarify",
        "what do you mean", "could you", "can we", "does", "is there"
    )
    if text.endswith("?") or text.startswith(question_starters):
        return "clarify"
    return "evaluate"


def handle_clarification_node(state: InterviewState) -> dict:
    from groq import Groq
    from dotenv import load_dotenv
    load_dotenv()

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    bar    = COMPANY_BAR.get(state["company"], DEFAULT_BAR)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are {bar['persona']} at {state['company']}. "
                    f"The candidate asked a clarifying question. Give a brief realistic hint — "
                    f"do NOT give away the answer or approach. 2-3 sentences max."
                )
            },
            {"role": "assistant", "content": state["current_question"]},
            {"role": "user",      "content": state["user_answer"]}
        ],
        temperature=0.5,
        max_tokens=150,
        stream=False
    )

    return {
        "clarification_response": response.choices[0].message.content.strip(),
        "user_answer": ""
    }


@traceable(name="evaluate_answer")
def evaluate_answer_node(state: InterviewState) -> dict:
    evaluation = evaluate_answer(
        question=state["current_question"],
        answer=state["user_answer"],
        round_type=state["round_type"],
        topic=state["topic"]
    )

    score = evaluation.get("score", 5)
    updated_scores = state.get("session_scores", []) + [{
        "topic": state["topic"],
        "score": score,
        "question": state["current_question"]
    }]

    return {
        "last_evaluation":        evaluation,
        "current_score":          score,
        "session_scores":         updated_scores,
        "clarification_response": "",
        "messages":               [{"role": "user", "content": state["user_answer"]}]
    }


def build_question_graph():
    graph = StateGraph(InterviewState)
    graph.add_node("retrieve_context",  retrieve_context_node)
    graph.add_node("generate_question", generate_question_node)
    graph.set_entry_point("retrieve_context")
    graph.add_edge("retrieve_context",  "generate_question")
    graph.add_edge("generate_question", END)
    return graph.compile()


def build_answer_graph():
    graph = StateGraph(InterviewState)
    graph.add_node("route_answer",         route_answer_node)
    graph.add_node("handle_clarification", handle_clarification_node)
    graph.add_node("evaluate_answer",      evaluate_answer_node)
    graph.set_entry_point("route_answer")
    graph.add_conditional_edges(
        "route_answer",
        is_clarification_or_answer,
        {
            "clarify":  "handle_clarification",
            "evaluate": "evaluate_answer"
        }
    )
    graph.add_edge("handle_clarification", END)
    graph.add_edge("evaluate_answer",      END)
    return graph.compile()


def create_initial_state(company: str, role: str, round_type: str,
                         topic: str, user_id: str = "default_user") -> InterviewState:
    return {
        "company":                company,
        "role":                   role,
        "round_type":             round_type,
        "topic":                  topic,
        "user_id":                user_id,
        "messages":               [],
        "current_question":       "",
        "current_difficulty":     "easy",
        "current_examples":       [],
        "rag_context":            [],
        "current_score":          0,
        "session_scores":         [],
        "question_number":        0,
        "max_questions":          5,
        "user_answer":            "",
        "last_evaluation":        {},
        "clarification_response": "",
        "asked_questions":        []
    }


if __name__ == "__main__":
    q_graph = build_question_graph()
    a_graph = build_answer_graph()
    state   = create_initial_state("Amazon", "SWE", "DSA", "arrays")

    state = q_graph.invoke(state)
    print(f"Q{state['question_number']} [{state['current_difficulty']}]:")
    print(state['current_question'])
    print(f"Examples: {state['current_examples']}")

    state["user_answer"] = "I would use a two pointer approach."
    state = a_graph.invoke(state)
    print(f"\nScore: {state['current_score']}")

    state = q_graph.invoke(state)
    print(f"\nQ{state['question_number']} [{state['current_difficulty']}]:")
    print(state['current_question'])