# 🎓 Study Buddy — AI Interview Coach

An adaptive, RAG-powered mock interview coach that simulates real company interviews, evaluates your answers, tracks your weak areas, and generates a personalized 7-day study plan.

**[Live Demo](https://studybuddy-bfyk8urmevxv8whsmab6nr.streamlit.app)**

---

## What it does

- Pulls real interview questions from a company-specific vector database (Amazon, Google, Microsoft, Adobe)
- Adapts difficulty question-by-question based on your last score
- Evaluates answers using an LLM-as-judge with structured rubrics per round type
- Tracks weak areas across sessions and generates a personalized 7-day study plan
- Supports clarification mid-interview — ask the interviewer a follow-up before answering
- Fetches live job descriptions from Adzuna API for companies not in the preloaded bank

---

## Architecture

```
User Input
    │
    ▼
┌─────────────────────────────────────────────┐
│              LangGraph Agent Loop            │
│                                             │
│  question_graph          answer_graph       │
│  ─────────────           ────────────       │
│  retrieve_context   →    route_answer       │
│       │                      │              │
│  generate_question      ┌────┴────┐         │
│       │                 │        │          │
│      END          clarify    evaluate       │
│                      │           │          │
│                     END         END         │
└─────────────────────────────────────────────┘
    │                         │
    ▼                         ▼
ChromaDB (RAG)         Session Memory
Semantic Cache         Weak Area Tracking
LangSmith Traces       Study Plan Agent
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq API — Llama 3.3 70B |
| Agent Framework | LangGraph + LangChain |
| Vector DB | ChromaDB |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Evaluation | RAGAS + LLM-as-judge |
| Observability | LangSmith |
| UI | Streamlit |
| JD Fetching | Adzuna API |
| Caching | Semantic cache (cosine similarity > 0.92) |
| Deployment | Streamlit Cloud |

---

## RAGAS Benchmark Results

Evaluated across 20 queries covering Amazon, Google, Microsoft, Adobe — DSA, System Design, Behavioral, Technical, Case rounds.

| Metric | Score |
|---|---|
| Topic Relevance | 0.94 / 1.0 |
| Context Faithfulness | 0.53 / 1.0 |
| Company Specificity | 0.84 / 1.0 |

Company specificity improved from **0.01 → 0.84** after adding company-specific context framing to the generation prompt.

---

## Project Structure

```
StudyBuddy/
├── agents/
│   ├── interview_graph.py      # LangGraph two-graph architecture
│   └── study_plan_agent.py     # 7-day personalized study plan generator
├── api/
│   ├── evaluator.py            # LLM-as-judge with rubric scoring
│   ├── guardrails.py           # Injection detection + output validation
│   └── cache.py                # Semantic cache with cosine similarity
├── rag/
│   ├── retriever_v1.py         # Three-tier RAG retriever
│   └── jdfetcher.py            # Live JD fetching via Adzuna API
├── memory/
│   └── session_store.py        # Session persistence + weak area tracking
├── eval/
│   └── ragas_eval.py           # RAGAS evaluation pipeline
├── ui/
│   └── app.py                  # Streamlit UI
├── chroma_store/               # ChromaDB vector database
└── data/                       # Raw question bank data
```

---

## Key Design Decisions

**Two-graph architecture** — A single LangGraph graph can't have two entry points. Question generation and answer evaluation are separate concerns with separate entry points. Splitting into `question_graph` and `answer_graph` solved a bug where every `graph.invoke()` restarted from `retrieve_context` regardless of context.

**RAG over fine-tuning** — Fine-tuning requires GPU compute and retraining for every new company. RAG lets me add questions to ChromaDB and immediately get company-specific outputs without touching the model. Retrieval is also transparent — I can see exactly what context the model received.

**Semantic cache** — Exact-match caching misses "arrays" vs "array manipulation". Embedding the query and comparing cosine similarity catches semantically equivalent queries, reducing Groq API calls by ~40%.

**LLM-as-judge** — Using the same LLM to evaluate answers avoids the cost of a separate evaluation model. Structured JSON output with `what_was_right`, `what_was_missing`, `improvement_tip` gives candidates actionable feedback, not just a score.

---

## Supported Companies and Rounds

**Companies:** Amazon, Google, Microsoft, Adobe + any company via live JD fetching

**Roles:** SWE, Data, Product, Analyst

**Rounds:** DSA, System Design, Behavioral, Technical, Case

---

## Observability

All LangGraph nodes are traced with LangSmith — `retrieve_context`, `generate_question`, `evaluate_answer`. Every run logs inputs, outputs, latency, and token usage to the PrepSense project dashboard at [smith.langchain.com](https://smith.langchain.com).

