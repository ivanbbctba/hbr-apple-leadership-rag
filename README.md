# hbr-apple-leadership-rag

**Production-grade RAG System with Multi-Mode Comparison, Evaluation & Cost Tracking**

A clean, well-architected Retrieval-Augmented Generation (RAG) project that demonstrates senior-level engineering practices in building LLM-powered systems. The project systematically compares **Raw LLM**, **Prompt Engineering**, and **RAG** approaches while incorporating evaluation, token tracking, and cost awareness.

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C?logo=langchain&logoColor=white)](https://python.langchain.com/)
[![RAG](https://img.shields.io/badge/RAG-Production-ff6b6b)](https://python.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-gpt-5.4-mini-412991?logo=openai&logoColor=white)](https://openai.com/)
[![Chroma](https://img.shields.io/badge/Chroma-Vector%20DB-FF6B6B)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-ff4b4b?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Pytest](https://img.shields.io/badge/Pytest-Testing-0A9F3C?logo=pytest&logoColor=white)](https://pytest.org/)
[![Pydantic](https://img.shields.io/badge/Pydantic-Settings-0A7E3C?logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

---

**Core Thesis**: In production AI systems, *how* you engineer the pipeline matters more than the model itself.

## Why This Project Matters

Most RAG portfolios only show “it works”. This project goes further by:

- Making the value of RAG **visible and measurable** through side-by-side comparison
- Adding **LLM-as-Judge evaluation** (Faithfulness + Relevance)
- Tracking **tokens and cost** per mode
- Maintaining clean separation of concerns and testability

This is the kind of project that signals you understand **systems engineering with AI**, not just prompting.

## Key Features

- **Three-Mode Comparison Engine**: Raw LLM vs Prompt Engineering vs RAG
- **LLM-as-Judge Evaluation**: Automatic scoring (0–5) for faithfulness and relevance
- **Token Usage & Cost Tracking**: Per-mode visibility with configurable pricing
- **Production-Ready Architecture**: Cached retriever, Pydantic settings, clean interfaces
- **Interactive Demo**: Streamlit UI with transparent context and evaluation scores
- **Solid Test Suite**: Unit + integration tests with mocking

## Architectural Decisions & Trade-offs

| Decision | Rationale | Trade-off |
|---------|---------|---------|
| **Per-mode `ResponseEngine`** (instead of single dispatcher) | Makes the comparison educational and explicit | Slightly more repetitive code |
| **LLM-as-Judge instead of RAGAS (initially)** | Faster iteration + lower cost during development | Less standardized than RAGAS |
| **Cached retriever (`lru_cache`)** | Avoids expensive re-initialization on every comparison | Small memory overhead |
| **Separate `judge_model`** | Allows using cheaper/faster model for evaluation | Adds one more configuration |
| **Explicit `ModeResult` dataclass** | Clear contract between layers + easy to extend | Slightly more boilerplate |

These decisions reflect real production thinking: balancing **demo clarity**, **cost**, **maintainability**, and **evaluation quality**.

## 🛠️ Tech Stack & Core Pipeline

**Core Technologies**
- **Language & Runtime**: Python 3.12+
- **LLM & Embeddings**: OpenAI (`gpt-4o-mini` for generation, `text-embedding-3-small` for retrieval)
- **RAG Orchestration**: LangChain
- **Vector Database**: Chroma (persistent local store)
- **Document Parsing & Chunking**: PyMuPDF + `RecursiveCharacterTextSplitter` (256 tokens, 25 overlap)
- **User Interface**: Streamlit (interactive 3-mode comparison dashboard)
- **Configuration**: `pydantic-settings` + `.env`
- **Testing**: pytest (unit + integration with mocking)
- **Package Management**: pipenv

**Key Engineering Patterns**
- RAG with source grounding
- Prompt Engineering (strict system instructions for business analysts)
- LLM-as-Judge evaluation (faithfulness + relevance scoring)
- Token usage & cost observability
- Cached retriever + configurable MMR retrieval
- Clean separation of concerns (`ResponseEngine`, `ComparisonEngine`, `Evaluation`, `MetricsCollector`)
- Structured `ModeResult` dataclass for clear contracts between layers

**Core Pipeline**
Ingestion → Chunking → Embeddings → Chroma vector store → MMR/Similarity retrieval (`k=3` + configurable MMR params) → Augmented prompt with strict grounding instructions → `gpt-4o-mini` generation → Evaluation + cost tracking.

---

## 🚀 Quick Start (Local)

```bash
git clone https://github.com/ivanbbctba/hbr-apple-leadership-rag.git
cd hbr-apple-leadership-rag

pipenv --python 3.12
pipenv install
pipenv install --dev black ruff pytest mypy pre-commit

cp .env.example .env
# Add your OpenAI API key

# Run the core pipeline
pipenv run python -m hbr_apple_rag.rag_pipeline

# Launch the interactive Streamlit dashboard
pipenv run streamlit run app.py

# Run tests
pipenv run pytest
```

---

## 📁 Project Structure

```
hbr-apple-leadership-rag/
├── app.py                          # Streamlit
├── src/
│   └── hbr_apple_rag/
│       ├── init.py
│       ├── config.py               # Pydantic settings
│       ├── comparison.py               # Orchestrator + cost + evaluation
│       ├── evaluation.py               # LLM-as-Judge
│       ├── metrics.py                  # Timing collection
│       ├── prompts.py              # System prompts + questions
│       ├── response_engine.py      # Core class (Raw / Prompt Eng / RAG)
│       └── rag_pipeline.py         # Document ingestion & vector store
├── data/
│   ├── raw/                        # Source PDFs
│   └── vector_store/               # Persisted Chroma database
├── tests/
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_comparison.py
│   ├── test_evaluation.py
│   ├── test_metrics.py
│   └── test_response_engine.py
├── Pipfile
├── Pipfile.lock
├── .env.example
├── pytest.ini
└── README.md
```

---
## 🔄 From Week-1 Notebook Prototype to Production System

This project began as a focused Week 1 exploration in a Jupyter notebook. That artifact implemented a functional RAG pipeline (LangChain + OpenAI + strict business-analyst system prompt) and performed a manual three-way comparison (raw LLM vs. prompt engineering vs. RAG) against three fixed questions on the HBR article *"How Apple Is Organized for Innovation"*. It correctly demonstrated that retrieval grounding produces materially better factual accuracy and source alignment than either baseline.

However, a notebook has structural limits when the goal is a portfolio piece that signals senior AI engineering capability:

- Monolithic cell-based code with high cognitive load for maintenance or extension  
- Qualitative comparison only (no automated faithfulness, relevance, cost, or latency signals)  
- Zero interactivity — fixed questions only, no way for a stakeholder to test ad-hoc queries  
- Missing production fundamentals: cost accounting, observability, graceful error handling, and testability

**On `feature/improve-streamlit` we executed a deliberate architectural lift.**

**Modular domain model**  
Introduced clear separation of concerns: `ResponseEngine` (unified interface for the three modes + `@lru_cache` retriever), `ComparisonEngine` (orchestration and aggregation), `MetricsCollector`, and an `Evaluation` module using LLM-as-Judge for faithfulness and relevance. Each component owns one responsibility. This directly improves testability (expanded `tests/` suite with pytest + mocks), reduces future change risk, and makes it trivial to add new modes, judges, or retrievers later. The original notebook had none of this structure.

**Interactive Streamlit UI — the pivotal portfolio decision**  
Chose to invest in a full Streamlit dashboard rather than stopping at CLI or notebook output. This was intentional. A live three-column comparison with custom question support (600-character limit), expandable retrieved context with chunk-level breakdown, per-mode token usage + estimated cost, response-time metrics, and automated LLM-judge scores transforms a theoretical result into an experience. 

**Observability and pragmatic evaluation layer**  
Added end-to-end tracking of tokens, USD cost (flexible model pricing), wall time, plus `evaluate_faithfulness` and `evaluate_relevance` via configurable judge model. LLM-as-Judge was selected as the pragmatic starting point — it requires no labeled dataset and directly operationalizes the grounding objective from the original notebook's system prompt. In production RAG this is table stakes: you must answer "is this hallucinated?", "how much did it cost?", and "was retrieval actually useful?". The notebook had none of these signals.

**Retrieval and robustness hardening**  
Configurable MMR parameters (`retriever_fetch_k`, `retriever_lambda_mult`) via pydantic-settings give explicit control over diversity vs. relevance. Unified error handling and structured `ModeResult` objects (including error state) ensure the demo never silently fails — essential for reliable portfolio sharing.

The static results table below preserves the original notebook validation on the three fixed questions. The live Streamlit app generalizes and quantifies that comparison: any user can now run those exact questions (or new ones) and see automated faithfulness/relevance scores and cost/latency data that were unavailable in the original notebook.

---

## 📊 Results: Raw LLM vs Prompt Engineering vs RAG

| Question | Raw LLM | Prompt Engineering | **RAG (Ours)** | Winner |
|---------|---------|--------------------|----------------|--------|
| **Q1: Authors & Publisher** | Failed to answer (asked for the article) | Failed to answer (asked for the article) | **Correctly identified** Joel M. Podolny and Morten T. Hansen, published in *Harvard Business Review* | **RAG** |
| **Q2: 3 Leadership Characteristics** | Generic leadership traits (Vision, Communication, Integrity) | Good structure but partially generic | **More aligned** with the article (Deep expertise, Immersion in details, Collaborative debate) | **RAG** |
| **Q3: Apple’s Leadership & Innovation** | Generic Apple examples (iPhone, iPod, MacBook) | Structured but lacked article grounding | **Strongly grounded** in the article: elimination of general managers, functional organization aligning expertise with decisions, collaborative debate culture, and strategic delegation/focus | **RAG** |

**Key Insight**:  
RAG significantly outperformed both Raw LLM and Prompt Engineering, especially on factual and source-specific questions. The retrieval mechanism allowed the model to ground its answers in the actual document, reducing hallucinations and improving relevance.

## 🧠 Conclusion

This project demonstrates the clear advantage of **Retrieval-Augmented Generation (RAG)** over both raw LLM usage and prompt engineering alone when working with dense, factual business documents.

While Prompt Engineering improves structure and tone, it still lacks access to the source material, often leading to generic or hallucinated answers. RAG, by combining strong prompting with relevant retrieved context, produces answers that are more accurate, specific, and grounded in the original article.

The results from the three original questions show that RAG consistently delivered higher quality, source-aligned responses — particularly on factual questions (such as authorship and publication details) and on extracting nuanced concepts from the text.

This project reinforces a key principle in modern LLM application development: **context is king**. Providing the model with the right information at inference time remains one of the most effective ways to improve reliability and reduce hallucinations in production systems.

---

## 🗺️ Roadmap & Next Engineering Steps

- [x] Modular architecture + `ResponseEngine` / `ComparisonEngine` separation
- [x] LLM-as-Judge evaluation (faithfulness + relevance)
- [x] Token usage, cost tracking & metrics collection
- [x] Interactive Streamlit demo with custom questions + context transparency
- [x] Comprehensive test suite (pytest + mocks)
- [ ] Guardrails for production-ready configuration
- [ ] Docker + docker-compose for reproducible environments
- [ ] GitHub Actions (lint, test, security scan)
- [ ] Pre-commit hooks + mypy strict typing
- [ ] Optional: RAGAS integration or hybrid judge + human eval loop
- [ ] Optional: multi-agent extensions (building on current comparison engine)

---

## 📝 License

GPL-3.0 — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

Harvard Business Review article: *"How Apple Is Organized for Innovation"* by Joel M. Podolny and Morten T. Hansen.  
Developed as part of the Postgraduate Program in Agentic AI for Business at UT Austin McCombs School of Business.

---

**Built by Ivan Beira** — Portfolio project demonstrating production-grade RAG engineering and business insight extraction.