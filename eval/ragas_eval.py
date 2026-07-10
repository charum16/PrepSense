import os
import sys
import json
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
from dotenv import load_dotenv
from rag.retriever_v1 import retrieve_questions
from groq import Groq
 
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
 
# ── Eval dataset ───────────────────────────────────────────────────────────────
# 20 queries covering different companies, roles, rounds, and topics.
# Each entry is: what a user would type into PrepSense.
EVAL_QUERIES = [
    # Amazon SWE DSA
    {"company": "Amazon", "role": "SWE", "round": "DSA", "topic": "arrays",             "ground_truth": "A question about array manipulation requiring hash maps or two pointers"},
    {"company": "Amazon", "role": "SWE", "round": "DSA", "topic": "dynamic programming","ground_truth": "A DP question about optimal substructure, like knapsack or longest subsequence"},
    {"company": "Amazon", "role": "SWE", "round": "DSA", "topic": "trees",              "ground_truth": "A binary tree traversal or BST question"},
    {"company": "Amazon", "role": "SWE", "round": "DSA", "topic": "graphs",             "ground_truth": "A graph traversal question using BFS or DFS"},
    {"company": "Amazon", "role": "SWE", "round": "DSA", "topic": "strings",            "ground_truth": "A string manipulation question involving sliding window or hash map"},
 
    # Amazon SWE System Design
    {"company": "Amazon", "role": "SWE", "round": "System Design", "topic": "distributed systems", "ground_truth": "A system design question about scalability and distributed architecture"},
    {"company": "Amazon", "role": "SWE", "round": "Behavioral",    "topic": "leadership",           "ground_truth": "A behavioral question testing Amazon Leadership Principles"},
 
    # Google SWE
    {"company": "Google", "role": "SWE", "round": "DSA", "topic": "binary search",      "ground_truth": "A binary search question on sorted arrays or search space reduction"},
    {"company": "Google", "role": "SWE", "round": "DSA", "topic": "dynamic programming","ground_truth": "A DP question about memoization or tabulation"},
    {"company": "Google", "role": "SWE", "round": "System Design", "topic": "search",   "ground_truth": "A system design question about search infrastructure at scale"},
 
    # Microsoft SWE
    {"company": "Microsoft", "role": "SWE", "round": "DSA", "topic": "linked lists",    "ground_truth": "A linked list manipulation question"},
    {"company": "Microsoft", "role": "SWE", "round": "DSA", "topic": "trees",           "ground_truth": "A tree traversal or balancing question"},
 
    # Adobe SWE
    {"company": "Adobe", "role": "SWE", "round": "DSA", "topic": "arrays",              "ground_truth": "An array manipulation question relevant to image processing"},
    {"company": "Adobe", "role": "SWE", "round": "DSA", "topic": "dynamic programming", "ground_truth": "A DP question about optimization"},
 
    # Generic fallback
    {"company": "Generic", "role": "SWE",     "round": "DSA",      "topic": "sorting",         "ground_truth": "A sorting algorithm question"},
    {"company": "Generic", "role": "Data",    "round": "Technical", "topic": "SQL",             "ground_truth": "A SQL query question with joins or window functions"},
    {"company": "Generic", "role": "Product", "round": "Case",      "topic": "product metrics", "ground_truth": "A product metrics or success criteria question"},
    {"company": "Generic", "role": "Analyst", "round": "Technical", "topic": "SQL",             "ground_truth": "A SQL aggregation or analytics question"},
 
    # Cross-topic (tests if retriever stays on topic)
    {"company": "Amazon", "role": "SWE", "round": "DSA", "topic": "heap",               "ground_truth": "A heap or priority queue question"},
    {"company": "Amazon", "role": "SWE", "round": "DSA", "topic": "sliding window",     "ground_truth": "A sliding window question on arrays or strings"},
]
 
 
# ── Step 1: Generate question using retriever ──────────────────────────────────
def generate_question_for_eval(company: str, role: str, round_type: str, topic: str) -> tuple:
    """
    Returns (retrieved_context, generated_question) for a given query.
    This mirrors what interview_graph.py does so RAGAS measures the real pipeline.
    """
    # Retrieve context
    rag_results = retrieve_questions(company, role, round_type, topic, n_results=5)
    context_str = "\n".join([
        f"- [{r['difficulty']}] {r['question']}" for r in rag_results
    ]) if rag_results else "No context available."
 
    # Generate question (same prompt as interview_graph.py)
    system = f"""You are a senior interviewer at {company} hiring for a {role} role.
 
Here are real {company} {round_type} questions from our verified database:
{context_str}
 
Your new question must match this difficulty and style. Frame it in {company}'s real engineering context — reference {company}'s actual products, systems, or scale where possible.
 
Respond in JSON only:
{{
    "question": "...",
    "difficulty": "medium",
    "topic": "{topic}",
    "examples": []
}}"""
 
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Ask a medium {round_type} question about {topic} framed in a real {company} engineering context — reference {company}'s actual systems or products where possible."}
        ],
        temperature=0.7,
        max_tokens=400,
        stream=False
    )
 
    raw = response.choices[0].message.content
    try:
        import json as _json
        result = _json.loads(raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
        generated_question = result.get("question", raw)
    except Exception:
        generated_question = raw
 
    return context_str, generated_question, rag_results
 
 
# ── Step 2: Build eval dataset ─────────────────────────────────────────────────
def build_eval_dataset() -> list:
    """
    Runs all 20 queries through the pipeline and records:
    - query (topic + company + role)
    - retrieved_contexts (what RAG returned)
    - response (generated question)
    - ground_truth (what we expect)
    """
    print(f"\n{'='*60}")
    print(f"Building RAGAS eval dataset — {len(EVAL_QUERIES)} queries")
    print(f"{'='*60}\n")
 
    dataset = []
 
    for i, q in enumerate(EVAL_QUERIES):
        print(f"[{i+1}/{len(EVAL_QUERIES)}] {q['company']} | {q['role']} | {q['round']} | {q['topic']}")
 
        try:
            context, generated, rag_results = generate_question_for_eval(
                q["company"], q["role"], q["round"], q["topic"]
            )
 
            dataset.append({
                "query":              f"{q['company']} {q['role']} {q['round']}: {q['topic']}",
                "company":            q["company"],
                "role":               q["role"],
                "round":              q["round"],
                "topic":              q["topic"],
                "retrieved_contexts": [context],
                "response":           generated,
                "ground_truth":       q["ground_truth"],
                "rag_source":         rag_results[0]["source"] if rag_results else "none",
                "n_results":          len(rag_results)
            })
            print(f"  ✓ Generated: {generated[:80]}...")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            dataset.append({
                "query":              f"{q['company']} {q['role']} {q['round']}: {q['topic']}",
                "company":            q["company"],
                "role":               q["role"],
                "round":              q["round"],
                "topic":              q["topic"],
                "retrieved_contexts": [""],
                "response":           "",
                "ground_truth":       q["ground_truth"],
                "rag_source":         "error",
                "n_results":          0
            })
 
    return dataset
 
 
# ── Step 3: Run RAGAS metrics ──────────────────────────────────────────────────
def run_ragas(dataset: list) -> dict:
    """
    Runs RAGAS faithfulness and context_precision on the dataset.
    Returns scores per query and overall averages.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, context_precision
        from datasets import Dataset
    except ImportError:
        print("\n⚠ RAGAS not installed. Run: pip install ragas datasets")
        return {}
 
    print(f"\n{'='*60}")
    print("Running RAGAS evaluation...")
    print(f"{'='*60}\n")
 
    # RAGAS expects a HuggingFace Dataset with specific column names
    hf_dataset = Dataset.from_list([
        {
            "question":           row["query"],
            "contexts":           row["retrieved_contexts"],
            "answer":             row["response"],
            "ground_truth":       row["ground_truth"],
        }
        for row in dataset
    ])
 
    try:
        result = evaluate(
            dataset=hf_dataset,
            metrics=[faithfulness, context_precision]
        )
        return result
    except Exception as e:
        print(f"RAGAS evaluation error: {e}")
        print("Falling back to manual scoring...")
        return {}
 
 
# ── Step 4: Manual scoring (fallback + supplement to RAGAS) ───────────────────
def manual_score(dataset: list) -> list:
    """
    For each generated question, use LLM-as-judge to score:
    1. Topic relevance (0-1): is the question actually about the requested topic?
    2. Context faithfulness (0-1): does the question match the difficulty/style of retrieved context?
    3. Company specificity (0-1): does it feel like a real company-specific question?
    """
    print(f"\n{'='*60}")
    print("Running manual LLM-as-judge scoring...")
    print(f"{'='*60}\n")
 
    scored = []
 
    for i, row in enumerate(dataset):
        if not row["response"]:
            scored.append({**row, "topic_relevance": 0, "context_faithfulness": 0, "company_specificity": 0})
            continue
 
        prompt = f"""You are evaluating an AI interview question generator.
 
Query: Generate a {row['round']} question about "{row['topic']}" for a {row['company']} {row['role']} interview.
 
Retrieved context from {row['company']}'s database:
{row['retrieved_contexts'][0][:500]}
 
Generated question:
{row['response']}
 
Score on 3 dimensions (each 0.0 to 1.0):
 
1. topic_relevance: Is the generated question actually about "{row['topic']}"?
   - 1.0 = perfectly on topic
   - 0.5 = loosely related
   - 0.0 = completely different topic (e.g. asked about trees when topic was arrays)
 
2. context_faithfulness: Does the generated question match the difficulty and style of the retrieved {row['company']} context?
   - 1.0 = same complexity level, similar problem structure
   - 0.5 = somewhat similar
   - 0.0 = ignores context, much easier or harder
 
3. company_specificity: Does this feel like a real {row['company']} interview question specifically?
   Consider: Does it reference {row['company']}'s products, scale, or engineering culture?
   For example — Amazon questions should feel like they involve AWS, Prime, delivery systems, or LP principles.
   Google questions should involve Search, Maps, YouTube scale problems.
   Microsoft questions should involve Azure, Office, Teams context.
   - 1.0 = clearly {row['company']}-specific, couldn't be from another company
   - 0.5 = generic but acceptable for {row['company']}
   - 0.0 = completely generic, no {row['company']} flavor at all
 
Be strict and vary your scores. Do NOT give the same score to every question.
 
Respond in JSON only:
{{"topic_relevance": 0.0, "context_faithfulness": 0.0, "company_specificity": 0.0, "reasoning": "specific explanation referencing the actual question content"}}"""
 
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are an evaluation expert. Respond in JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200,
                stream=False
            )
            import json as _json
            scores = _json.loads(
                response.choices[0].message.content.strip()
                    .removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            )
            scored.append({**row, **scores})
            print(f"[{i+1}/{len(dataset)}] {row['topic']:20s} | relevance={scores.get('topic_relevance', 0):.1f} | faithfulness={scores.get('context_faithfulness', 0):.1f} | specificity={scores.get('company_specificity', 0):.1f}")
        except Exception as e:
            print(f"[{i+1}/{len(dataset)}] {row['topic']:20s} | Error: {e}")
            scored.append({**row, "topic_relevance": 0, "context_faithfulness": 0, "company_specificity": 0})
 
    return scored
 
 
# ── Step 5: Print report ───────────────────────────────────────────────────────
def print_report(scored: list, ragas_result: dict = None):
    print(f"\n{'='*60}")
    print("RAGAS EVALUATION REPORT")
    print(f"{'='*60}\n")
 
    if ragas_result:
        print("RAGAS Metrics:")
        for k, v in ragas_result.items():
            print(f"  {k}: {v:.3f}")
        print()
 
    # Manual scores
    if scored:
        avg_relevance     = sum(r.get("topic_relevance", 0) for r in scored) / len(scored)
        avg_faithfulness  = sum(r.get("context_faithfulness", 0) for r in scored) / len(scored)
        avg_specificity   = sum(r.get("company_specificity", 0) for r in scored) / len(scored)
 
        print("Manual LLM-as-Judge Scores (avg):")
        print(f"  Topic Relevance:       {avg_relevance:.2f} / 1.0")
        print(f"  Context Faithfulness:  {avg_faithfulness:.2f} / 1.0")
        print(f"  Company Specificity:   {avg_specificity:.2f} / 1.0")
 
        # Worst performers
        print(f"\n{'─'*60}")
        print("WORST PERFORMING RETRIEVALS (fix these in retriever_v1.py):")
        print(f"{'─'*60}")
        worst = sorted(scored, key=lambda x: x.get("topic_relevance", 0))[:5]
        for w in worst:
            print(f"\n  Company:  {w['company']} | Role: {w['role']} | Topic: {w['topic']}")
            print(f"  Source:   {w.get('rag_source', 'unknown')} ({w.get('n_results', 0)} results)")
            print(f"  Scores:   relevance={w.get('topic_relevance', 0):.1f} | faithfulness={w.get('context_faithfulness', 0):.1f}")
            print(f"  Question: {w.get('response', '')[:100]}...")
            print(f"  Reason:   {w.get('reasoning', 'N/A')}")
 
        # Best performers
        print(f"\n{'─'*60}")
        print("BEST PERFORMING RETRIEVALS:")
        print(f"{'─'*60}")
        best = sorted(scored, key=lambda x: x.get("topic_relevance", 0), reverse=True)[:3]
        for b in best:
            print(f"\n  Company:  {b['company']} | Topic: {b['topic']}")
            print(f"  Source:   {b.get('rag_source', 'unknown')}")
            print(f"  Scores:   relevance={b.get('topic_relevance', 0):.1f} | faithfulness={b.get('context_faithfulness', 0):.1f}")
 
    # RAG source breakdown
    print(f"\n{'─'*60}")
    print("RAG SOURCE BREAKDOWN:")
    sources = {}
    for r in scored:
        s = r.get("rag_source", "unknown")
        sources[s] = sources.get(s, 0) + 1
    for s, count in sources.items():
        print(f"  {s}: {count} queries")
 
 
# ── Step 6: Save results ───────────────────────────────────────────────────────
def save_results(scored: list, ragas_result: dict = None):
    os.makedirs("eval", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"eval/ragas_results_{timestamp}.json"
 
    output = {
        "timestamp":    timestamp,
        "n_queries":    len(scored),
        "ragas_scores": ragas_result if ragas_result else {},
        "manual_scores": {
            "avg_topic_relevance":    round(sum(r.get("topic_relevance", 0) for r in scored) / len(scored), 3) if scored else 0,
            "avg_context_faithfulness": round(sum(r.get("context_faithfulness", 0) for r in scored) / len(scored), 3) if scored else 0,
            "avg_company_specificity":  round(sum(r.get("company_specificity", 0) for r in scored) / len(scored), 3) if scored else 0,
        },
        "results": scored
    }
 
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
 
    print(f"\n✓ Results saved to {path}")
    return path
 
 
# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ragas", action="store_true", help="Skip RAGAS metrics (faster, uses manual scoring only)")
    parser.add_argument("--quick",      action="store_true", help="Run only first 5 queries for quick testing")
    args = parser.parse_args()
 
    if args.quick:
        EVAL_QUERIES = EVAL_QUERIES[:5]
        print("Quick mode: running 5 queries only")
 
    # Step 1: Build dataset
    dataset = build_eval_dataset()
 
    # Step 2: RAGAS metrics (optional — requires OpenAI key or local LLM)
    ragas_result = {}
    if not args.skip_ragas:
        ragas_result = run_ragas(dataset)
 
    # Step 3: Manual LLM-as-judge scoring
    scored = manual_score(dataset)
 
    # Step 4: Print report
    print_report(scored, ragas_result)
 
    # Step 5: Save
    save_results(scored, ragas_result)
 