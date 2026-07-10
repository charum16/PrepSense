
from groq import Groq
from dotenv import load_dotenv
import os
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.guardrails import safe_evaluate, validate_eval_output

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

RUBRICS = {
    "DSA": [
        "Correct algorithm identified",
        "Time complexity analyzed",
        "Space complexity analyzed",
        "Edge cases mentioned",
        "Code structure and readability"
    ],
    "System Design": [
        "Requirements clarified",
        "High-level architecture proposed",
        "Key components identified",
        "Scalability considered",
        "Trade-offs discussed"
    ],
    "Behavioral": [
        "Situation clearly described (S)",
        "Task and responsibility stated (T)",
        "Action taken explained (A)",
        "Result and impact shared (R)"
    ],
    "Technical": [
        "Core concept correctly explained",
        "Practical application mentioned",
        "Edge cases or limitations discussed"
    ],
    "Case": [
        "Problem clearly structured",
        "Hypothesis-driven approach",
        "Data/metrics identified",
        "Recommendation clearly stated"
    ]
}


def _raw_evaluate(question: str, answer: str, round_type: str, topic: str) -> dict:
    """
    Core evaluation logic — unchanged from original.
    Called by safe_evaluate which handles validation + retry.
    """
    rubric     = RUBRICS.get(round_type, RUBRICS["Technical"])
    rubric_str = "\n".join([f"- {r}" for r in rubric])

    prompt = f"""You are evaluating a candidate's interview answer.

Question: {question}

Candidate's answer: {answer}

Evaluation rubric for {round_type} round:
{rubric_str}

Think step by step:
1. Check which rubric points the candidate covered
2. Identify what was missing or weak
3. Assign a score from 1-10

Respond in this exact JSON format only:
{{
    "score": <integer 1-10>,
    "what_was_right": ["point 1", "point 2"],
    "what_was_missing": ["point 1", "point 2"],
    "improvement_tip": "one specific actionable tip",
    "star_complete": <true or false>
}}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a strict but fair technical interviewer. Always respond in valid JSON only."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.3,
        max_tokens=600,
        stream=False
    )

    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except Exception:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return json.loads(cleaned)
        except Exception:
            return {
                "score": 0,
                "what_was_right": [],
                "what_was_missing": ["Could not parse evaluation"],
                "improvement_tip": "Please try again.",
                "star_complete": False
            }


def evaluate_answer(question: str, answer: str, round_type: str, topic: str) -> dict:
    """
    Public interface — wraps _raw_evaluate with guardrail validation + retry.
    Always returns a valid dict with all required fields.
    """
    return safe_evaluate(_raw_evaluate, question, answer, round_type, topic)


if __name__ == "__main__":
    result = evaluate_answer(
        question="Find two numbers in an array that sum to a target.",
        answer="I would use a hash map to store complements. For each number, check if target minus that number exists in the map. O(n) time, O(n) space.",
        round_type="DSA",
        topic="arrays"
    )
    print(json.dumps(result, indent=2))