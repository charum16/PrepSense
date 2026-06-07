##PrepSense — AI Mock Interview Coach

The interview prep tool that actually knows what your target company is looking for.

Most interview prep tools give you generic LeetCode questions and canned feedback. PrepSense is different — it reads real job descriptions, conducts adaptive mock interviews grounded in what each company actually wrote, evaluates your answers with a rubric-based LLM judge, and remembers your weak areas across every session so it gets smarter about you over time.

The Problem
Every year, millions of candidates fail interviews not because they're unqualified — but because they prepared for the wrong things. Generic prep platforms don't know that Amazon cares about ownership, that Google values system reliability at scale, or that Meta wants data-driven product thinkers. Candidates walk in underprepared for the specific role, specific company, and specific bar.
PrepSense fixes that.

What It Does
A user picks a role and target company. PrepSense:

Retrieves the most relevant chunks from that company's job description using semantic search
Generates an adaptive interview question grounded in what the JD actually says — not a generic textbook question
Evaluates the user's answer in real-time using an LLM-as-judge with a structured rubric (completeness, accuracy, communication clarity)
Remembers weak areas across sessions — so next time, it pushes harder on exactly what you struggled with
Generates a personalized 7-day study plan based on your performance history

The result: an interview experience that feels like a real interviewer who has read your target company's JD and remembers every session you've had.

Architecture
User picks role + company
        ↓
Session Agent — loads user history from Postgres
        ↓
RAG Retriever — semantic search over JD corpus in ChromaDB
        ↓
Question Generator Agent (ReAct loop) — grounded question from JD context
        ↓
User answers
        ↓
Evaluator Agent (LLM-as-judge) — structured JSON score + feedback
        ↓
Postgres — stores topic scores + weak areas
        ↓
Next session: harder questions on weak topics
4 agents. 2 knowledge bases. 1 persistent memory layer.

Tech Stack
LayerTechnologyLLMClaude Sonnet (Anthropic API)Agent frameworkLangGraphEmbeddingssentence-transformers (all-MiniLM-L6-v2)Vector DBChromaDB → Pinecone at scalePersistent memoryPostgreSQLRAG evaluationRAGAS (faithfulness + context precision)ObservabilityLangSmithBackendFastAPIFrontendStreamlit

Key Engineering Decisions
Why LLM-as-judge instead of keyword matching?
Rubric-based semantic evaluation is significantly more reliable on open-ended responses. The evaluator agent returns structured JSON with per-dimension scores — making it debuggable and explainable, not a black box.
Why ChromaDB + semantic chunking?
Fixed-size chunking on JDs produced noisy retrieval because JD boilerplate was diluting the signal. Sentence-aware chunking with metadata filters (role, company, difficulty) dramatically improved retrieval precision — measurable with RAGAS faithfulness scores.
Why Postgres for long-term memory instead of just the vector DB?
Full transcript storage causes context bloat. Instead, PrepSense stores a compressed representation of user performance per topic — retrieved at session start to personalize without inflating context length.
Why LangGraph for the agent loop?
The question generator agent can loop if it can't find a relevant question in the retrieved context. LangGraph's explicit node-edge graph gives deterministic control over loop termination and fallback paths — critical for production reliability.

RAG Evaluation Results
PrepSense is benchmarked with RAGAS on a 20-question evaluation set:
MetricBaselineAfter optimizationFaithfulness0.580.81Context Precision0.610.79
Optimization involved switching from fixed-size to sentence-aware chunking and adding role + company metadata filters.
