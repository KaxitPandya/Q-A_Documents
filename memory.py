"""
Document Q&A Assistant — Conversation Memory with Progressive Summarization

When chat history exceeds a threshold, older messages are *summarized*
by the LLM instead of being hard-truncated. This preserves long-range
context (e.g. "the document we discussed earlier") while keeping prompt
size manageable and cost low.

Flow:
  history ≤ threshold  →  use verbatim
  history >  threshold  →  summarize old messages + keep recent verbatim
"""

from typing import Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from logger import logger


# ═════════════════════════════════════════════════════════════
# Summarization prompt
# ═════════════════════════════════════════════════════════════
_SUMMARIZE_PROMPT = ChatPromptTemplate.from_template(
    """\
Summarize the following conversation between a user and an AI document
assistant. Preserve:
  • Key facts and data points discussed
  • Which documents / topics were covered
  • Any conclusions or follow-up questions

Be concise (3-5 sentences) but don't lose critical context.

Conversation:
{conversation}

Summary:"""
)


# ═════════════════════════════════════════════════════════════
# Progressive summarization
# ═════════════════════════════════════════════════════════════
def summarize_history(
    messages: List[Dict],
    llm: ChatOpenAI,
    *,
    keep_recent: int = 4,
    existing_summary: str = "",
) -> str:
    """
    Progressive conversation summarization.

    If the conversation is short (≤ keep_recent messages), return it
    verbatim. Otherwise, summarize older messages and prepend the
    summary to the recent verbatim messages.

    Args:
        messages:         Full conversation history
        llm:              LLM for summarization
        keep_recent:      Number of recent messages to always keep verbatim
        existing_summary: Previous summary (for incremental summarization)

    Returns:
        Formatted chat history string with [Summary] + [Recent]
    """
    if len(messages) <= keep_recent:
        return _format_messages(messages)

    # Split into old (to summarize) and recent (to keep)
    old_messages = messages[:-keep_recent]
    recent_messages = messages[-keep_recent:]

    # Build text to summarize (include existing summary if any)
    old_text = ""
    if existing_summary:
        old_text = f"[Previous summary]\n{existing_summary}\n\n[New messages]\n"
    old_text += _format_messages(old_messages)

    # Summarize
    chain = _SUMMARIZE_PROMPT | llm | StrOutputParser()
    try:
        summary = chain.invoke({"conversation": old_text[:4000]}).strip()
        logger.info(
            "Summarized %d old messages → %d chars",
            len(old_messages),
            len(summary),
        )
    except Exception as exc:
        logger.warning("Conversation summarization failed: %s", exc)
        # Fallback: just use recent messages
        return _format_messages(recent_messages)

    # Combine summary + recent verbatim
    recent_text = _format_messages(recent_messages)
    return (
        f"[Earlier conversation summary]\n{summary}\n\n"
        f"[Recent messages]\n{recent_text}"
    )


def should_summarize(messages: List[Dict], threshold: int = 8) -> bool:
    """Check if the conversation is long enough to benefit from summarization."""
    return len(messages) > threshold


def _format_messages(messages: List[Dict]) -> str:
    """Format messages list into a readable string."""
    parts = []
    for msg in messages:
        prefix = "User" if msg["role"] == "user" else "Assistant"
        content = msg.get("raw_answer", msg.get("content", ""))
        parts.append(f"{prefix}: {content}")
    return "\n".join(parts)
