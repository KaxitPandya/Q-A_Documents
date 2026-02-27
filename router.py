"""
Document Q&A Assistant — Adaptive RAG Router + Self-RAG

Adaptive RAG:  Classifies query complexity → auto-selects pipeline depth.
  no_retrieval → greetings, chitchat, basic math (Self-RAG)
  simple       → one-hop factual → vector+BM25 only
  moderate     → multi-part or reasoning → hybrid + CRAG
  complex      → multi-hop, comparison → full pipeline + grounding

Self-RAG:  Before any retrieval, decide if retrieval is even needed.
  If the question is a greeting, math, or meta-question, skip retrieval
  entirely and answer directly — saving latency and cost.

References:
  • Adaptive RAG   — Jeong et al. 2024 (https://arxiv.org/abs/2403.14403)
  • Self-RAG       — Asai et al. 2023 (https://arxiv.org/abs/2310.11511)
"""

from typing import Any, Dict
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from logger import logger


# ═════════════════════════════════════════════════════════════
# 1. Query Complexity Classifier
# ═════════════════════════════════════════════════════════════
_ROUTER_PROMPT = ChatPromptTemplate.from_template(
    """\
You are a query complexity classifier for a document Q&A system.
The user has uploaded documents and is asking questions about them.

Classify the query into EXACTLY one category:

- "no_retrieval" — greetings, chitchat, simple math, meta-questions about the system,
  or anything that does NOT need document lookup.
  Examples: "hello", "what is 2+2", "thank you", "how does this app work?"

- "simple" — straightforward factual lookup that a single document chunk can answer.
  Examples: "What is the company's revenue?", "Who is the CEO?", "When was it founded?"

- "moderate" — needs synthesis from multiple parts, summarization, or some reasoning.
  Examples: "Summarize the key findings", "What are the main risks mentioned?",
            "List all the recommendations"

- "complex" — multi-hop reasoning, cross-referencing, comparison, deep analysis,
  or likely needs external knowledge beyond the uploaded documents.
  Examples: "How does the strategy in section 2 relate to the risks in section 5?",
            "Compare the financial performance across all quarters and identify trends",
            "What are the implications of these findings for the broader industry?"

Return ONLY one word: no_retrieval, simple, moderate, or complex.

Query: {question}
Classification:"""
)


# ═════════════════════════════════════════════════════════════
# 2. Route Configurations (pipeline settings per complexity)
# ═════════════════════════════════════════════════════════════
ROUTE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "no_retrieval": {
        "use_hybrid": False,
        "use_reranking": False,
        "use_crag": False,
        "use_grounding": False,
        "use_expansion": False,
        "use_reformulation": False,
        "k": 0,
        "model": "gpt-4o-mini",
        "description": "⚡ Direct answer — no retrieval needed",
        "emoji": "⚡",
    },
    "simple": {
        "use_hybrid": True,
        "use_reranking": False,
        "use_crag": False,
        "use_grounding": False,
        "use_expansion": False,
        "use_reformulation": True,
        "k": 3,
        "model": "gpt-4o-mini",
        "description": "🔍 Simple lookup — hybrid search",
        "emoji": "🔍",
    },
    "moderate": {
        "use_hybrid": True,
        "use_reranking": False,
        "use_crag": True,
        "use_grounding": False,
        "use_expansion": False,
        "use_reformulation": True,
        "k": 5,
        "model": "gpt-4o-mini",
        "description": "🧠 Moderate — hybrid + Corrective RAG",
        "emoji": "🧠",
    },
    "complex": {
        "use_hybrid": True,
        "use_reranking": True,
        "use_crag": True,
        "use_grounding": True,
        "use_expansion": True,
        "use_reformulation": True,
        "k": 8,
        "model": "gpt-4o",
        "description": "🔬 Complex — full pipeline + grounding",
        "emoji": "🔬",
    },
}


# ═════════════════════════════════════════════════════════════
# 3. Classify & Route
# ═════════════════════════════════════════════════════════════
def classify_query(question: str, llm: ChatOpenAI) -> str:
    """
    LLM-based query complexity classification.
    Returns: "no_retrieval", "simple", "moderate", or "complex".
    """
    chain = _ROUTER_PROMPT | llm | StrOutputParser()
    try:
        result = chain.invoke({"question": question}).strip().lower().split()[0]
        # Clean up potential punctuation
        result = result.strip(".,!?;:'\"")
        if result not in ROUTE_CONFIGS:
            logger.warning("Unknown route '%s', defaulting to moderate", result)
            result = "moderate"
    except Exception as exc:
        logger.warning("Query classification failed (%s), defaulting to moderate", exc)
        result = "moderate"

    logger.info("Query classified → '%s': %s", result, question[:80])
    return result


def route_query(
    question: str,
    model_name: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    Adaptive RAG Router entry point.

    1. Classify the query complexity (Self-RAG check included)
    2. Return the optimal pipeline configuration

    Returns dict with:
      - route: str (no_retrieval|simple|moderate|complex)
      - description: str
      - use_hybrid, use_reranking, use_crag, use_grounding, etc.
    """
    llm = ChatOpenAI(model=model_name, temperature=0.0)
    route = classify_query(question, llm)
    config = {**ROUTE_CONFIGS[route], "route": route}

    logger.info("Routed to '%s': %s", route, config["description"])
    return config
