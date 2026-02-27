"""
Document Q&A Assistant — Advanced Retrieval Pipeline
Implements:
  • Hybrid Search  (BM25 keyword + vector semantic, fused with RRF)
  • LLM Re-ranking (score & re-order retrieved chunks)
  • Corrective RAG  (grade retrieval quality → decide next action)
  • Reciprocal Rank Fusion (merge ranked lists from different retrievers)
"""

import hashlib
from typing import Any, Dict, List, Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from logger import logger


# ═════════════════════════════════════════════════════════════
# 1. BM25 Retriever  (keyword / sparse)
# ═════════════════════════════════════════════════════════════
class BM25Retriever:
    """
    Lightweight Okapi BM25 retriever over in-memory documents.
    Uses rank_bm25 for scoring.
    """

    def __init__(self, documents: List[Document], k: int = 5):
        from rank_bm25 import BM25Okapi

        self.documents = documents
        self.k = k
        # Tokenize
        self._corpus = [doc.page_content.lower().split() for doc in documents]
        self._bm25 = BM25Okapi(self._corpus)

    def invoke(self, query: str) -> List[Document]:
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        # Top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.k]
        return [self.documents[i] for i in top_indices if scores[i] > 0]


# ═════════════════════════════════════════════════════════════
# 2. Reciprocal Rank Fusion (RRF)
# ═════════════════════════════════════════════════════════════
def reciprocal_rank_fusion(
    ranked_lists: List[List[Document]],
    k: int = 60,
) -> List[Document]:
    """
    Merge multiple ranked lists using RRF.
    Score(doc) = Σ 1 / (k + rank_i)  across all lists.
    Returns documents sorted by fused score (highest first).
    """
    fused_scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            doc_id = hashlib.md5(doc.page_content.encode()).hexdigest()
            doc_map[doc_id] = doc
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    sorted_ids = sorted(fused_scores, key=lambda d: fused_scores[d], reverse=True)
    results = []
    for doc_id in sorted_ids:
        doc = doc_map[doc_id]
        doc.metadata["rrf_score"] = round(fused_scores[doc_id], 6)
        results.append(doc)

    return results


# ═════════════════════════════════════════════════════════════
# 3. Hybrid Search  (vector + BM25 → RRF fusion)
# ═════════════════════════════════════════════════════════════
def hybrid_search(
    query: str,
    vector_store,
    all_chunks: List[Document],
    *,
    k: int = 5,
    retrieval_mode: str = "similarity",
    fetch_k: int = 10,
) -> List[Document]:
    """
    Run both vector retrieval AND BM25 keyword retrieval, then
    fuse results with Reciprocal Rank Fusion.
    This catches documents that are semantically similar (vector)
    AND those that share exact keywords (BM25) — far more robust
    than either approach alone.
    """
    # --- Vector retrieval ---
    search_kwargs: Dict[str, Any] = {"k": k}
    if retrieval_mode == "mmr":
        search_kwargs["fetch_k"] = fetch_k
        search_kwargs["lambda_mult"] = 0.5

    vector_results = vector_store.as_retriever(
        search_type=retrieval_mode,
        search_kwargs=search_kwargs,
    ).invoke(query)

    # --- BM25 retrieval ---
    try:
        bm25 = BM25Retriever(all_chunks, k=k)
        bm25_results = bm25.invoke(query)
    except Exception as exc:
        logger.warning("BM25 retrieval failed (%s), using vector-only.", exc)
        bm25_results = []

    # --- Fuse ---
    fused = reciprocal_rank_fusion([vector_results, bm25_results])
    logger.info(
        "Hybrid search: vector=%d, bm25=%d → fused=%d",
        len(vector_results), len(bm25_results), len(fused),
    )
    return fused[:k]


# ═════════════════════════════════════════════════════════════
# 4. LLM Re-ranking
# ═════════════════════════════════════════════════════════════
_RERANK_PROMPT = ChatPromptTemplate.from_template(
    """\
You are a relevance grader. Given a user question and a document chunk,
score how relevant the chunk is to answering the question.
Return ONLY a number from 0 to 10. Nothing else.

Question: {question}
Document chunk: {chunk}

Relevance score (0-10):"""
)


def llm_rerank(
    question: str,
    documents: List[Document],
    llm: ChatOpenAI,
    top_n: int = 5,
) -> List[Document]:
    """
    Use the LLM to score each chunk's relevance 0-10, then return
    the top_n most relevant. More precise than raw vector distance.
    """
    chain = _RERANK_PROMPT | llm | StrOutputParser()
    scored: List[Tuple[float, Document]] = []

    for doc in documents:
        try:
            raw = chain.invoke({
                "question": question,
                "chunk": doc.page_content[:800],  # truncate to save tokens
            })
            score = float(raw.strip().split()[0])
        except (ValueError, IndexError):
            score = 5.0  # default mid-score on parse failure
        except Exception as exc:
            logger.warning("Rerank scoring failed for chunk: %s", exc)
            score = 5.0

        doc.metadata["rerank_score"] = score
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    reranked = [doc for _, doc in scored[:top_n]]
    logger.info(
        "LLM re-ranking: %d → top %d (scores: %s)",
        len(documents),
        len(reranked),
        [s for s, _ in scored[:top_n]],
    )
    return reranked


# ═════════════════════════════════════════════════════════════
# 5. Corrective RAG  (CRAG)
# ═════════════════════════════════════════════════════════════
_GRADE_PROMPT = ChatPromptTemplate.from_template(
    """\
You are a grader assessing whether a set of retrieved documents is
sufficient to answer the user's question.

Question: {question}

Retrieved context (first 1500 chars):
{context_preview}

Grade the retrieval as one of:
- "good"    — the context clearly contains info to answer the question
- "partial" — some relevant info but incomplete
- "poor"    — the context is mostly irrelevant

Return ONLY one word: good, partial, or poor."""
)


def grade_retrieval(
    question: str,
    documents: List[Document],
    llm: ChatOpenAI,
) -> str:
    """
    Corrective RAG step: assess if retrieved docs are good enough.
    Returns: "good", "partial", or "poor".
    """
    context = "\n\n".join(d.page_content[:500] for d in documents[:3])
    chain = _GRADE_PROMPT | llm | StrOutputParser()

    try:
        grade = chain.invoke({
            "question": question,
            "context_preview": context[:1500],
        }).strip().lower()
        if grade not in ("good", "partial", "poor"):
            grade = "partial"
    except Exception as exc:
        logger.warning("Retrieval grading failed: %s", exc)
        grade = "partial"

    logger.info("CRAG grade for '%s': %s", question[:60], grade)
    return grade


# ═════════════════════════════════════════════════════════════
# 6. Contextual Question Reformulation
# ═════════════════════════════════════════════════════════════
_REFORMULATE_PROMPT = ChatPromptTemplate.from_template(
    """\
Given the chat history and the latest user question, reformulate the
question so it is fully self-contained (no pronouns like "it", "that",
"they" referring to previous context). If the question is already
standalone, return it as-is.

Chat history:
{chat_history}

Latest question: {question}

Standalone question:"""
)


def reformulate_question(
    question: str,
    chat_history: str,
    llm: ChatOpenAI,
) -> str:
    """
    Rewrite a follow-up question so retrieval works without needing
    the conversation context. E.g. "What about its CEO?" →
    "Who is the CEO of Tesla?"
    """
    if not chat_history.strip():
        return question

    chain = _REFORMULATE_PROMPT | llm | StrOutputParser()
    try:
        reformulated = chain.invoke({
            "question": question,
            "chat_history": chat_history,
        }).strip()
        if reformulated:
            logger.info("Reformulated: '%s' → '%s'", question[:60], reformulated[:60])
            return reformulated
    except Exception as exc:
        logger.warning("Question reformulation failed: %s", exc)

    return question


# ═════════════════════════════════════════════════════════════
# 7. Hallucination / Grounding Check
# ═════════════════════════════════════════════════════════════
_GROUNDING_PROMPT = ChatPromptTemplate.from_template(
    """\
You are a fact-checker. Determine if the Answer is fully supported
by the provided Context. Return ONLY one word:

- "supported"    — every claim in the answer is backed by the context
- "partial"      — some claims are supported, some are not
- "unsupported"  — the answer contains fabricated information

Context:
{context}

Answer:
{answer}

Verdict:"""
)


def check_grounding(
    answer: str,
    documents: List[Document],
    llm: ChatOpenAI,
) -> str:
    """
    Post-generation check: is the answer grounded in retrieved context?
    Returns "supported", "partial", or "unsupported".
    """
    context = "\n\n".join(d.page_content[:500] for d in documents[:4])
    chain = _GROUNDING_PROMPT | llm | StrOutputParser()

    try:
        verdict = chain.invoke({
            "context": context[:2000],
            "answer": answer[:1000],
        }).strip().lower()
        if verdict not in ("supported", "partial", "unsupported"):
            verdict = "partial"
    except Exception as exc:
        logger.warning("Grounding check failed: %s", exc)
        verdict = "partial"

    logger.info("Grounding check: %s", verdict)
    return verdict
