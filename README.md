# 📚 Document Q&A Assistant — Advanced Adaptive RAG Pipeline

> **Live demo:** [q-adocuments.streamlit.app](https://q-adocuments.streamlit.app/)

An AI-powered **Retrieval-Augmented Generation (RAG)** system with an **Adaptive Router** that auto-selects the optimal pipeline depth per query — featuring Self-RAG, streaming answers, conversation summarization, semantic chunking, hybrid search, LLM re-ranking, Corrective RAG, hallucination detection, and autonomous agentic reasoning.

Built with **LangChain LCEL** · **OpenAI** · **ChromaDB** · **Streamlit**

---

## 🧠 What Makes This Different

Most RAG apps use a static pipeline: embed → vector search → generate. This project implements **Adaptive RAG** — the system *intelligently routes* each query through the optimal pipeline:

```
┌───────────────────────────────────────────────────────────────────┐
│                    🧭 ADAPTIVE RAG ROUTER                        │
│  LLM classifies query complexity → selects pipeline depth        │
│                                                                   │
│  "hello"            → no_retrieval  (Self-RAG: skip retrieval)   │
│  "What is revenue?" → simple        (hybrid search only)         │
│  "Summarize risks"  → moderate      (hybrid + CRAG)              │
│  "Compare Ch2 & 5"  → complex       (full pipeline + grounding)  │
└────────────────────────────┬──────────────────────────────────────┘
                             │
┌────────────────────────────▼──────────────────────────────────────┐
│  1. SELF-RAG CHECK                                                │
│     Does this query even need retrieval?                          │
│     Greetings, math, meta-questions → answer directly             │
├───────────────────────────────────────────────────────────────────┤
│  2. QUESTION REFORMULATION                                        │
│     Rewrite follow-ups ("what about its CEO?") into               │
│     self-contained queries for better retrieval                   │
├───────────────────────────────────────────────────────────────────┤
│  3. SMART CHUNKING  (3 strategies)                                │
│     • Recursive (fast, character-based)                           │
│     • Semantic  (embedding-based boundary detection)              │
│     • Parent-Child (small chunks for retrieval,                   │
│       large parents for context)                                  │
├───────────────────────────────────────────────────────────────────┤
│  4. HYBRID SEARCH  (BM25 + Vector → RRF)                         │
│     Keyword retrieval (BM25 Okapi) + semantic retrieval           │
│     fused with Reciprocal Rank Fusion                             │
├───────────────────────────────────────────────────────────────────┤
│  5. LLM RE-RANKING                                                │
│     LLM scores each chunk 0-10 for relevance                     │
├───────────────────────────────────────────────────────────────────┤
│  6. CORRECTIVE RAG (CRAG)                                         │
│     Grade retrieval: good / partial / poor                        │
│     If poor → automatic Wikipedia fallback                        │
├───────────────────────────────────────────────────────────────────┤
│  7. STREAMING ANSWER GENERATION                                   │
│     Token-by-token streaming via LCEL chain.stream()              │
├───────────────────────────────────────────────────────────────────┤
│  8. HALLUCINATION GROUNDING CHECK                                 │
│     Post-generation: is the answer supported by context?          │
├───────────────────────────────────────────────────────────────────┤
│  9. CONVERSATION MEMORY                                           │
│     Progressive summarization — older messages are LLM-summarized │
│     instead of truncated, preserving long-range context           │
└───────────────────────────────────────────────────────────────────┘
```

### Three Pipeline Modes

| Mode | Description |
|------|-------------|
| **🧠 Adaptive** (default) | Auto-classifies each query and selects the right pipeline depth. Simple → fast. Complex → full pipeline. |
| **🤖 Agent** | Autonomous tool-calling agent (OpenAI function calling) that decides whether to search documents, search Wikipedia, or answer directly. |
| **🔧 Manual** | Full manual control over every pipeline stage for power users. |

---

## ✨ Feature List

| Category | Feature | Description |
|----------|---------|-------------|
| **Router** | Adaptive RAG | LLM classifies query → auto-selects pipeline depth |
| **Router** | Self-RAG | Skips retrieval entirely for greetings, math, meta-questions |
| **Chunking** | Semantic chunking | Splits at *meaning* boundaries using embedding cosine similarity |
| **Chunking** | Parent-child | Small chunks for retrieval, large parents for LLM context |
| **Retrieval** | Hybrid search | BM25 keyword + vector semantic, fused via RRF |
| **Retrieval** | LLM re-ranking | LLM scores each chunk's relevance 0-10 |
| **Retrieval** | Query expansion | Multi-query: LLM rephrases question 2 ways for broader recall |
| **Quality** | Corrective RAG | Grades retrieval, falls back to Wikipedia if poor |
| **Quality** | Grounding check | Detects hallucination: is answer supported by context? |
| **Quality** | Confidence scoring | Multi-signal: word overlap + CRAG grade + grounding |
| **UX** | Streaming answers | Token-by-token display via LCEL chain.stream() |
| **UX** | Conversation memory | Progressive LLM summarization of older messages |
| **UX** | Pipeline visibility | Expandable "pipeline steps" shows exactly what ran |
| **UX** | Route badges | Visual indicator of which route was selected per query |
| **Agent** | Tool-calling agent | Autonomous reasoning: documents, Wikipedia, direct answer |
| **Infra** | Retry with backoff | Exponential retry for all OpenAI calls |
| **Infra** | Token & cost tracking | Live tokens + USD estimate in sidebar |

---

## 🏗️ Architecture

```
app.py                 ← Streamlit UI & orchestration
│
├── router.py          ← Adaptive RAG Router + Self-RAG classifier
├── memory.py          ← Conversation summarization (progressive)
├── chunking.py        ← SemanticChunker, ParentChildChunker, recursive
├── retrieval.py       ← BM25, RRF fusion, re-ranking, CRAG, grounding
├── qa_chain.py        ← RAG pipeline (prepare_context + stream/invoke)
├── agent.py           ← Tool-calling RAG agent (OpenAI functions)
├── vector_store.py    ← ChromaDB embedding with retry
├── document_loader.py ← File loaders (PDF, DOCX, TXT, MD, CSV, Wikipedia)
├── config.py          ← All settings, model configs
└── logger.py          ← Structured logging
```

### Key Design Decisions

- **Adaptive routing** replaces manual presets — the system is smarter about cost/latency/quality tradeoffs than the user
- **Streaming** splits the pipeline into `prepare_rag_context()` (blocking retrieval) + `generate_answer_stream()` (streaming generation)
- **Progressive summarization** uses the LLM to compress older conversations instead of hard-truncating at N messages
- **Self-RAG** (no-retrieval route) saves latency and cost for questions that don't need document lookup

---

## 🚀 Quick Start

### Local development

```bash
git clone https://github.com/<your-username>/Q-A_Documents.git
cd Q-A_Documents

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
streamlit run app.py
```

### Streamlit Cloud (production)

The app is deployed at [q-adocuments.streamlit.app](https://q-adocuments.streamlit.app/).
API key is configured via Streamlit Secrets (`openai_api_key`).

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | OpenAI GPT-4o-mini / GPT-4o |
| Embeddings | text-embedding-3-small |
| Vector DB | ChromaDB |
| Keyword search | BM25 Okapi (rank_bm25) |
| Framework | LangChain LCEL + Agents |
| Frontend | Streamlit |
| Language | Python 3.10+ |

---

## 📁 Project Structure

```
.
├── app.py               # Streamlit UI — main entry point
├── router.py            # Adaptive RAG Router + Self-RAG
├── memory.py            # Conversation summarization
├── chunking.py          # Semantic, parent-child, recursive chunking
├── retrieval.py         # Hybrid search, RRF, re-ranking, CRAG, grounding
├── qa_chain.py          # RAG pipeline (context prep + streaming)
├── agent.py             # Agentic RAG with tool calling
├── vector_store.py      # ChromaDB embedding & retry
├── document_loader.py   # File loading & preprocessing
├── config.py            # Configuration & constants
├── logger.py            # Structured logging
├── requirements.txt     # Dependencies
├── .gitignore
└── README.md
```

---

## 📚 Research References

This project implements concepts from several RAG research papers:

| Paper | Concept Used |
|-------|-------------|
| [Adaptive RAG (Jeong et al., 2024)](https://arxiv.org/abs/2403.14403) | Query complexity classification → adaptive pipeline |
| [Self-RAG (Asai et al., 2023)](https://arxiv.org/abs/2310.11511) | Decide when retrieval is needed |
| [Corrective RAG (Yan et al., 2024)](https://arxiv.org/abs/2401.15884) | Grade retrieval → corrective action |
| [Reciprocal Rank Fusion (Cormack et al., 2009)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) | Fuse multiple ranked lists |

---

## 📈 Future Enhancements

- [ ] RAGAS / DeepEval automated evaluation pipeline
- [ ] Open-source LLM support (Ollama) for local inference
- [ ] Graph RAG for relationship-heavy documents
- [ ] Multi-modal: image/table extraction from PDFs
- [ ] Excel / PowerPoint file support

---

## 📄 License

MIT
