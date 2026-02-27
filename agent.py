"""
Document Q&A Assistant — Agentic RAG

A tool-calling agent that *reasons* about which action to take:
  1. Search uploaded documents (vector + BM25 hybrid)
  2. Search Wikipedia for missing knowledge
  3. Answer directly from chat history / general knowledge
  4. Refuse if the question is unanswerable

Uses LangChain 1.x `create_agent` (LangGraph-based) — the agent
calls the LLM in a loop, invoking tools as needed, until no more
tool_calls remain.
"""

from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_agent

from logger import logger


# ─────────────────────────────────────────────────────────────
# Agent system prompt
# ─────────────────────────────────────────────────────────────
AGENT_SYSTEM_PROMPT = """\
You are an intelligent Document Q&A agent. You have access to tools.

Strategy:
1. ALWAYS try the document_search tool first — the user uploaded docs for a reason.
2. If document_search returns no useful results, try wikipedia_search.
3. If neither tool helps, say you don't have enough information — NEVER make up facts.
4. Cite your sources in the answer (document name, chunk number).
5. Be concise and accurate.
"""


# ─────────────────────────────────────────────────────────────
# Tool factories — we create tools that capture the live
# vector_store and chunk list via closure
# ─────────────────────────────────────────────────────────────
def _build_document_search_tool(vector_store, all_chunks: List[Document], k: int = 5):
    """Return a callable tool that searches the uploaded documents."""

    def document_search(query: str) -> str:
        """Search the user's uploaded documents for information relevant to the query.
        Use this tool first for any question about the uploaded content."""
        from retrieval import hybrid_search

        docs = hybrid_search(
            query, vector_store, all_chunks, k=k,
        )
        if not docs:
            return "No relevant information found in the uploaded documents."

        parts = []
        for i, doc in enumerate(docs, 1):
            src = doc.metadata.get("source", "unknown")
            ci = doc.metadata.get("chunk_index", "?")
            parts.append(f"[Source {i}: {src}, chunk {ci}]\n{doc.page_content[:600]}")
        return "\n\n".join(parts)

    return document_search


def _build_wikipedia_tool():
    """Return a callable tool that searches Wikipedia."""

    def wikipedia_search(query: str) -> str:
        """Search Wikipedia for general knowledge about a topic.
        Use this only if document_search didn't find enough information."""
        from langchain_community.document_loaders import WikipediaLoader

        try:
            loader = WikipediaLoader(query=query, load_max_docs=1)
            docs = loader.load()
            if docs:
                return docs[0].page_content[:2000]
            return "No Wikipedia article found."
        except Exception as exc:
            return f"Wikipedia search failed: {exc}"

    return wikipedia_search


# ─────────────────────────────────────────────────────────────
# Build and run the agent
# ─────────────────────────────────────────────────────────────
def build_agent(
    vector_store,
    all_chunks: List[Document],
    *,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.1,
    k: int = 5,
):
    """
    Create a LangChain 1.x agent (LangGraph-based) with
    document_search and wikipedia_search tools.

    Returns a compiled StateGraph that can be invoked.
    """
    llm = ChatOpenAI(model=model_name, temperature=temperature)

    tools = [
        _build_document_search_tool(vector_store, all_chunks, k=k),
        _build_wikipedia_tool(),
    ]

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=AGENT_SYSTEM_PROMPT,
    )

    logger.info("Agent built with %d tools (LangChain 1.x create_agent)", len(tools))
    return agent


def run_agent(
    question: str,
    agent,
    chat_history: Optional[List] = None,
) -> Dict[str, Any]:
    """
    Invoke the agent and return structured output.

    The new LangChain 1.x agent uses a messages-based interface:
      input:  {"messages": [list of messages]}
      output: {"messages": [list of messages incl. tool calls + final]}
    """
    # Build message history
    messages = []
    if chat_history:
        for msg in chat_history[-6:]:  # last 3 turns
            if msg["role"] == "user":
                messages.append({"role": "user", "content": msg["content"]})
            else:
                content = msg.get("raw_answer", msg.get("content", ""))
                messages.append({"role": "assistant", "content": content})

    # Add the current question
    messages.append({"role": "user", "content": question})

    result = agent.invoke({"messages": messages})

    # Extract final answer from the last AI message
    output_messages = result.get("messages", [])
    answer = ""
    tools_used = []
    steps = 0

    for msg in output_messages:
        # Count tool calls
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tools_used.append(tc.get("name", "unknown"))
                steps += 1
        # The last AIMessage without tool_calls is the final answer
        if hasattr(msg, "content") and isinstance(msg, AIMessage):
            if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                answer = msg.content

    return {
        "answer": answer or "I couldn't find a relevant answer.",
        "tools_used": tools_used,
        "steps": steps,
    }
