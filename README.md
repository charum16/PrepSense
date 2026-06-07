# PrepSense — AI-Powered Mock Interview Coach

PrepSense is an interview preparation platform that creates company-specific mock interviews using real job descriptions. Instead of asking generic interview questions, it tailors every session to the expectations of a particular role and company, then tracks performance over time to continuously adapt future interviews.

## Overview

Most interview preparation platforms focus on generic question banks and one-size-fits-all feedback. In reality, different companies evaluate candidates differently. A backend engineer interviewing at Amazon is expected to demonstrate different strengths than one interviewing at Google or Meta.

PrepSense bridges that gap by analyzing job descriptions, generating context-aware interview questions, evaluating responses using structured rubrics, and building a long-term profile of each user's strengths and weaknesses.

The result is a personalized interview coach that becomes more effective with every session.

---

## Problem Statement

Many candidates spend weeks preparing for interviews yet still struggle because their preparation is misaligned with the role they are targeting.

Traditional preparation platforms have several limitations:

- Generic question sets unrelated to specific job descriptions
- No understanding of company-specific expectations
- Limited personalization across sessions
- Static feedback that does not evolve with user progress

PrepSense addresses these issues by grounding interviews in real job descriptions and maintaining persistent memory of user performance.

---

## Features

### Job Description-Aware Question Generation

Relevant sections of a target company's job description are retrieved using semantic search. Questions are generated directly from the retrieved context, ensuring alignment with the role requirements.

### Adaptive Mock Interviews

Question difficulty evolves based on previous performance. Strong topics receive deeper follow-up questions, while weaker areas receive additional practice.

### LLM-Based Answer Evaluation

User responses are evaluated using a structured rubric covering:

- Completeness
- Technical accuracy
- Communication clarity

Scores and feedback are returned in a consistent JSON format for transparency and debugging.

### Long-Term Performance Memory

Instead of storing entire conversation histories, PrepSense maintains topic-level performance summaries in PostgreSQL. This allows personalization without excessive context growth.

### Personalized Study Plans

Based on historical interview performance, the platform generates targeted study recommendations and a customized 7-day preparation roadmap.

---

## System Architecture

```text
User
  │
  ▼
Session Agent
(Loads user history from PostgreSQL)
  │
  ▼
RAG Retriever
(ChromaDB semantic search over JD corpus)
  │
  ▼
Question Generator Agent
(JD-grounded interview questions)
  │
  ▼
User Response
  │
  ▼
Evaluator Agent
(LLM-as-Judge)
  │
  ▼
PostgreSQL
(Stores scores and weak topics)
  │
  ▼
Future Sessions
(Adaptive questioning based on performance)
```

---

## Tech Stack

| Component | Technology |
|------------|------------|
| LLM | Claude Sonnet (Anthropic API) |
| Agent Orchestration | LangGraph |
| Embeddings | all-MiniLM-L6-v2 |
| Vector Database | ChromaDB |
| Scalable Vector Store | Pinecone |
| Persistent Memory | PostgreSQL |
| RAG Evaluation | RAGAS |
| Observability | LangSmith |
| Backend | FastAPI |
| Frontend | Streamlit |

---

## Engineering Decisions

### LLM-as-Judge Evaluation

Traditional keyword matching struggles with open-ended interview responses.

PrepSense uses rubric-based evaluation to score answers across multiple dimensions and return structured feedback. This approach provides greater flexibility while remaining explainable and easy to audit.

### Semantic Chunking for Job Descriptions

Initial experiments with fixed-size chunking produced noisy retrieval results because boilerplate content frequently appeared in retrieved contexts.

Switching to sentence-aware chunking significantly improved retrieval quality. Additional metadata filtering based on role and company further increased precision.

### PostgreSQL for User Memory

Storing full conversation histories quickly increases context size and retrieval costs.

Instead, PrepSense stores compressed topic-level performance summaries, allowing personalization without introducing unnecessary context overhead.

### LangGraph for Agent Control

The question-generation workflow occasionally requires retries when retrieved context is insufficient.

LangGraph provides explicit graph-based control over execution flow, enabling deterministic retries, fallback paths, and termination conditions.

---

## RAG Evaluation

The retrieval pipeline was evaluated using RAGAS on a 20-question benchmark set.

| Metric | Baseline | Optimized |
|----------|----------|-----------|
| Faithfulness | 0.58 | 0.81 |
| Context Precision | 0.61 | 0.79 |

### Improvements Applied

- Replaced fixed-size chunking with sentence-aware chunking
- Added role-based metadata filtering
- Added company-based metadata filtering
- Refined retrieval pipeline for higher semantic relevance

These changes produced substantial gains in both faithfulness and retrieval precision.

---

## Future Improvements

- Voice-based mock interviews
- Real-time interview simulations
- Multi-agent behavioral interview evaluation
- Company-specific competency frameworks
- Analytics dashboard for long-term progress tracking
- Support for system design and coding interview rounds

---

## Outcome

PrepSense combines retrieval-augmented generation, agent workflows, and persistent user memory to deliver interview practice tailored to individual goals. By grounding questions in real job descriptions and adapting over time, it provides a more realistic and targeted preparation experience than traditional interview practice platforms.
