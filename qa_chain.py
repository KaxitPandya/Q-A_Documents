"""
Document Q&A Assistant — QA Chain, Retrieval & Answer Generation

Full RAG pipeline split into two phases for streaming support:
  Phase 1 — prepare_rag_context():    stages 1-5 (retrieval side)
  Phase 2 — generate_answer_stream(): stage 6  (yields tokens)
           or ask_question():          full blocking pipeline

Pipeline stages:
  1. Contextual question reformulation (chat-history-aware)
  2. Hybrid search  (vector + BM25 → RRF fusion)
  3. Parent-document expansion
  4. LLM re-ranking
  5. Corrective RAG  (grade retrieval → fallback if poor)
  6. LCEL answer generation (streaming or blocking)
  7. Hallucination / grounding check
  8. Source citation
"""

import hashlib
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from retrieval import (
    hybrid_search,
    llm_rerank,
    grade_retrieval,
    reformulate_question,
    check_grounding,
)
from config import MAX_RETRIES, RETRY_BACKOFF, REQUEST_TIMEOUT, TOKEN_COSTS
from logger import logger


# ── Prompts ─────────────────────────────────────────────────
QA_SYSTEM_PROMPT = """\
You are a precise Document Q&A assistant.
Use ONLY the provided context to answer. If the context does not contain enough
information, say so clearly — never fabricate facts.

Rules:
- Be concise but thorough.
- Cite the source document name and chunk number when possible.
- If multiple documents are relevant, synthesise the answer from all of them.
- Structure longer answers with bullet points or numbered lists.
"""

QA_HUMAN_TEMPLATE = """\
Context:
{context}

Chat history:
{chat_history}

Question: {question}
"""

# Used when Self-RAG/Adaptive Router skips retrieval
DIRECT_SYSTEM_PROMPT = """\
You are a helpful assistant. The user is chatting in a Document Q&A app
but this particular message does not require document lookup.
Answer naturally and concisely.
"""

QUERY_EXPANSION_TEMPLATE = """\
Rephrase the following question in 2 different ways to improve document retrieval.
Return ONLY the rephrased questions, one per line.

Question: {question}
"""


def _format_docs(docs: List[Document]) -> str:
    """Join document chunks into a single context string with source tags."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        chunk = doc.metadata.get("chunk_index", "?")
        parts.append(f"[Source {i}: {source}, chunk {chunk}]\n{doc.page_content}")
    return "\n\n".join(parts)


# ── Token / Cost tracking ──────────────────────────────────
class TokenTracker:
    """Lightweight tracker that accumulates token usage across calls."""

    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_embedding_tokens = 0
        self.call_count = 0

    def record(self, prompt_tokens: int, completion_tokens: int):
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.call_count += 1

    def record_embedding(self, tokens: int):
        self.total_embedding_tokens += tokens

    def estimated_cost(self, model: str) -> float:
        costs = TOKEN_COSTS.get(model, {"input": 0, "output": 0})
        input_cost = (self.total_prompt_tokens / 1000) * costs["input"]
        output_cost = (self.total_completion_tokens / 1000) * costs["output"]
        return round(input_cost + output_cost, 6)

    def summary(self, model: str) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "embedding_tokens": self.total_embedding_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "api_calls": self.call_count,
            "estimated_cost_usd": self.estimated_cost(model),
        }


# ── Source highlighting ─────────────────────────────────────
def highlight_sources(answer: str, sources: List[Document]) -> str:
    """Markdown-friendly source citations appended to the answer."""
    if not sources:
        return answer

    seen = set()
    refs: List[str] = []
    for doc in sources:
        src = doc.metadata.get("source", "Unknown")
        chunk = doc.metadata.get("chunk_index", "?")
        key = f"{src}#{chunk}"
        if key not in seen:
            seen.add(key)
            snippet = doc.page_content[:120].replace("\n", " ")
            rerank = doc.metadata.get("rerank_score")
            score_str = f" | relevance {rerank}/10" if rerank is not None else ""
            refs.append(f"- **{src}** (chunk {chunk}{score_str}): _{snippet}…_")

    if refs:
        answer += "\n\n---\n📌 **Sources:**\n" + "\n".join(refs[:5])
    return answer


# ── Query expansion ─────────────────────────────────────────
def expand_query(question: str, llm: ChatOpenAI) -> List[str]:
    """Return original + 2 rephrased variants of *question*."""
    try:
        prompt = ChatPromptTemplate.from_template(QUERY_EXPANSION_TEMPLATE)
        chain = prompt | llm | StrOutputParser()
        raw = chain.invoke({"question": question})
        variants = [q.strip() for q in raw.strip().split("\n") if q.strip()]
        return [question] + variants[:2]
    except Exception:
        logger.warning("Query expansion failed; using original query.")
        return [question]


# ═════════════════════════════════════════════════════════════
# Phase 1: Prepare RAG context (stages 1-5)
# ═════════════════════════════════════════════════════════════
def prepare_rag_context(
    question: str,
    vector_store,
    *,
    all_chunks: Optional[List[Document]] = None,
    parent_map: Optional[Dict[str, Document]] = None,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.1,
    retrieval_mode: str = "similarity",
    k: int = 5,
    fetch_k: int = 10,
    use_hybrid: bool = True,
    use_reranking: bool = False,
    use_crag: bool = False,
    use_expansion: bool = False,
    use_reformulation: bool = True,
    chat_history: str = "",
    cache: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Run retrieval stages 1-5 of the RAG pipeline.

    Returns dict with:
      effective_question, relevant_docs, retrieval_grade,
      pipeline_steps, llm (for reuse in generation).
    """
    pipeline_steps: List[str] = []

    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        timeout=REQUEST_TIMEOUT,
    )

    # ── Stage 1: Contextual question reformulation ──────
    effective_question = question
    if use_reformulation and chat_history.strip():
        effective_question = reformulate_question(question, chat_history, llm)
        if effective_question != question:
            pipeline_steps.append(f"🔄 Reformulated → _{effective_question}_")

    # ── Stage 2: Retrieve ───────────────────────────────
    cache_key = hashlib.md5(
        f"{effective_question}|{retrieval_mode}|{k}|{use_hybrid}".encode()
    ).hexdigest()

    if cache is not None and cache_key in cache:
        relevant_docs = cache[cache_key]
        pipeline_steps.append("⚡ Cache hit")
    else:
        if use_hybrid and all_chunks:
            relevant_docs = hybrid_search(
                effective_question,
                vector_store,
                all_chunks,
                k=k,
                retrieval_mode=retrieval_mode,
                fetch_k=fetch_k,
            )
            pipeline_steps.append("🔍 Hybrid search (vector + BM25 → RRF)")
        else:
            search_kwargs: Dict[str, Any] = {"k": k}
            if retrieval_mode == "mmr":
                search_kwargs["fetch_k"] = fetch_k
                search_kwargs["lambda_mult"] = 0.5
            relevant_docs = vector_store.as_retriever(
                search_type=retrieval_mode,
                search_kwargs=search_kwargs,
            ).invoke(effective_question)
            pipeline_steps.append(f"🔍 Vector search ({retrieval_mode})")

        # Query expansion
        if use_expansion:
            queries = expand_query(effective_question, llm)
            seen_hashes = {
                hashlib.md5(d.page_content.encode()).hexdigest()
                for d in relevant_docs
            }
            for q in queries[1:]:
                extra = vector_store.as_retriever(
                    search_type=retrieval_mode,
                    search_kwargs={"k": k},
                ).invoke(q)
                for doc in extra:
                    h = hashlib.md5(doc.page_content.encode()).hexdigest()
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        relevant_docs.append(doc)
            pipeline_steps.append("📝 Query expansion")

        if cache is not None:
            cache[cache_key] = relevant_docs

    # ── Stage 3: Parent-document expansion ──────────────
    if parent_map:
        expanded = []
        seen_parents = set()
        for doc in relevant_docs:
            pid = doc.metadata.get("parent_id")
            if pid and pid in parent_map and pid not in seen_parents:
                seen_parents.add(pid)
                expanded.append(parent_map[pid])
            else:
                expanded.append(doc)
        relevant_docs = expanded
        pipeline_steps.append("📄 Parent-doc expansion")

    # ── Stage 4: LLM re-ranking ────────────────────────
    if use_reranking and len(relevant_docs) > 2:
        relevant_docs = llm_rerank(
            effective_question, relevant_docs, llm, top_n=min(k, len(relevant_docs))
        )
        pipeline_steps.append("🏆 LLM re-ranking")

    # ── Stage 5: Corrective RAG ────────────────────────
    retrieval_grade = "n/a"
    if use_crag:
        retrieval_grade = grade_retrieval(effective_question, relevant_docs, llm)
        pipeline_steps.append(f"✅ CRAG grade: {retrieval_grade}")

        if retrieval_grade == "poor":
            pipeline_steps.append("🌐 Fallback → Wikipedia search")
            try:
                from document_loader import load_wikipedia
                wiki_docs = load_wikipedia(effective_question, max_docs=1)
                if wiki_docs:
                    for wd in wiki_docs:
                        wd.metadata["source"] = "Wikipedia (CRAG fallback)"
                        wd.metadata["chunk_index"] = "wiki"
                    relevant_docs = wiki_docs + relevant_docs[:2]
            except Exception:
                logger.warning("CRAG Wikipedia fallback failed.")

    return {
        "effective_question": effective_question,
        "relevant_docs": relevant_docs,
        "retrieval_grade": retrieval_grade,
        "pipeline_steps": pipeline_steps,
        "llm": llm,
    }


# ═════════════════════════════════════════════════════════════
# Phase 2a: Stream answer tokens (for Streamlit streaming)
# ═════════════════════════════════════════════════════════════
def generate_answer_stream(
    effective_question: str,
    relevant_docs: List[Document],
    chat_history: str,
    llm: ChatOpenAI,
) -> Generator[str, None, None]:
    """
    Generator that yields answer tokens one at a time.
    Use with Streamlit's st.write_stream().

    Runs the full LCEL chain but streams the LLM output.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", QA_SYSTEM_PROMPT),
        ("human", QA_HUMAN_TEMPLATE),
    ])

    chain = (
        {
            "context": lambda _: _format_docs(relevant_docs),
            "chat_history": lambda _: chat_history,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    for token in chain.stream(effective_question):
        yield token


# ═════════════════════════════════════════════════════════════
# Phase 2b: Direct answer stream (Self-RAG: no retrieval)
# ═════════════════════════════════════════════════════════════
def direct_answer_stream(
    question: str,
    chat_history: str,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.3,
) -> Generator[str, None, None]:
    """
    Stream a direct answer withOUT any document retrieval.
    Used when the Adaptive Router classifies as 'no_retrieval' (Self-RAG).
    """
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        timeout=REQUEST_TIMEOUT,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", DIRECT_SYSTEM_PROMPT),
        ("human", "Chat history:\n{chat_history}\n\nQuestion: {question}"),
    ])

    chain = (
        {
            "chat_history": lambda _: chat_history,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    for token in chain.stream(question):
        yield token


# ═════════════════════════════════════════════════════════════
# Full blocking pipeline (non-streaming fallback)
# ═════════════════════════════════════════════════════════════
def ask_question(
    question: str,
    vector_store,
    *,
    all_chunks: Optional[List[Document]] = None,
    parent_map: Optional[Dict[str, Document]] = None,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.1,
    retrieval_mode: str = "similarity",
    k: int = 5,
    fetch_k: int = 10,
    use_hybrid: bool = True,
    use_reranking: bool = False,
    use_crag: bool = False,
    use_grounding: bool = False,
    use_expansion: bool = False,
    use_reformulation: bool = True,
    chat_history: str = "",
    cache: Optional[Dict] = None,
    token_tracker: Optional[TokenTracker] = None,
) -> Dict[str, Any]:
    """
    Full RAG pipeline (blocking / non-streaming).
    Returns dict with: answer, raw_answer, sources, confidence,
    response_time, retrieval_grade, grounding_verdict, pipeline_steps.
    """
    start = time.time()

    # ── Stages 1-5: Retrieval pipeline ──────────────────
    ctx = prepare_rag_context(
        question,
        vector_store,
        all_chunks=all_chunks,
        parent_map=parent_map,
        model_name=model_name,
        temperature=temperature,
        retrieval_mode=retrieval_mode,
        k=k,
        fetch_k=fetch_k,
        use_hybrid=use_hybrid,
        use_reranking=use_reranking,
        use_crag=use_crag,
        use_expansion=use_expansion,
        use_reformulation=use_reformulation,
        chat_history=chat_history,
        cache=cache,
    )

    relevant_docs = ctx["relevant_docs"]
    pipeline_steps = ctx["pipeline_steps"]
    llm = ctx["llm"]
    effective_question = ctx["effective_question"]
    retrieval_grade = ctx["retrieval_grade"]

    # ── Stage 6: Generate answer (LCEL + retry) ─────────
    prompt = ChatPromptTemplate.from_messages([
        ("system", QA_SYSTEM_PROMPT),
        ("human", QA_HUMAN_TEMPLATE),
    ])

    chain = (
        {
            "context": lambda _: _format_docs(relevant_docs),
            "chat_history": lambda _: chat_history,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            answer = chain.invoke(effective_question)
            break
        except Exception as exc:
            wait = RETRY_BACKOFF ** attempt
            logger.warning("Answer gen attempt %d/%d: %s", attempt, MAX_RETRIES, exc)
            if attempt == MAX_RETRIES:
                raise
            time.sleep(wait)

    pipeline_steps.append("💬 Answer generated")

    # ── Stage 7: Grounding / hallucination check ────────
    grounding_verdict = "n/a"
    if use_grounding and answer:
        grounding_verdict = check_grounding(answer, relevant_docs, llm)
        pipeline_steps.append(f"🛡️ Grounding: {grounding_verdict}")

        if grounding_verdict == "unsupported":
            answer = (
                "⚠️ **Warning:** The generated answer may not be fully supported by "
                "the retrieved documents. Please verify independently.\n\n" + answer
            )

    # ── Token tracking (best-effort) ────────────────────
    if token_tracker:
        try:
            md = getattr(llm, "_last_generation_info", None) or {}
            usage = md.get("token_usage", {})
            if usage:
                token_tracker.record(
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                )
        except Exception:
            pass

    # ── Confidence ──────────────────────────────────────
    confidence = _compute_confidence(
        relevant_docs, answer or "", retrieval_grade, grounding_verdict
    )

    answer_with_sources = highlight_sources(answer or "", relevant_docs)

    return {
        "answer": answer_with_sources,
        "raw_answer": answer or "",
        "sources": relevant_docs,
        "confidence": confidence,
        "response_time": round(time.time() - start, 2),
        "retrieval_grade": retrieval_grade,
        "grounding_verdict": grounding_verdict,
        "pipeline_steps": pipeline_steps,
        "effective_question": effective_question,
    }


def _compute_confidence(
    sources: List[Document],
    answer: str,
    retrieval_grade: str,
    grounding_verdict: str,
) -> float:
    """
    Multi-signal confidence score:
      - word overlap (baseline)
      - retrieval grade from CRAG
      - grounding verdict
    """
    if not sources or not answer:
        return 0.0

    answer_words = set(answer.lower().split()[:30])
    scores = []
    for doc in sources[:4]:
        content_words = set(doc.page_content.lower().split()[:150])
        overlap = len(answer_words & content_words)
        scores.append(overlap / max(len(answer_words), 1))
    word_score = min(sum(scores) / max(len(scores), 1), 1.0)

    grade_map = {"good": 0.15, "partial": 0.0, "poor": -0.2, "n/a": 0.0}
    grade_adj = grade_map.get(retrieval_grade, 0.0)

    ground_map = {"supported": 0.15, "partial": 0.0, "unsupported": -0.25, "n/a": 0.0}
    ground_adj = ground_map.get(grounding_verdict, 0.0)

    return max(0.0, min(1.0, word_score + grade_adj + ground_adj))
