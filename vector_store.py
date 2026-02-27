"""
Document Q&A Assistant — Vector Store Management
Handles ChromaDB creation, embedding, and retrieval with retry logic.
"""

import hashlib
import time
from typing import List, Optional

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
    CHROMA_PERSIST_DIR,
    MAX_RETRIES,
    RETRY_BACKOFF,
)
from logger import logger


def _get_embeddings() -> OpenAIEmbeddings:
    """Return the shared embedding function (modern model)."""
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS,
        chunk_size=1000,          # batch size for the API
        show_progress_bar=False,
    )


def create_vector_store(
    chunks: List[Document],
    collection_name: str = "default",
) -> Optional[Chroma]:
    """
    Embed *chunks* into ChromaDB with exponential-backoff retry.
    Returns the Chroma vector store or None on failure.
    """
    embeddings = _get_embeddings()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            vector_store = Chroma.from_documents(
                chunks,
                embeddings,
                collection_name=collection_name,
                persist_directory=CHROMA_PERSIST_DIR,
            )
            logger.info(
                "Vector store '%s' created (%d chunks, attempt %d)",
                collection_name,
                len(chunks),
                attempt,
            )
            return vector_store

        except Exception as exc:
            wait = RETRY_BACKOFF ** attempt
            logger.warning(
                "Embedding attempt %d/%d failed: %s — retrying in %ds",
                attempt,
                MAX_RETRIES,
                exc,
                wait,
            )
            if attempt == MAX_RETRIES:
                logger.error("All %d embedding attempts failed.", MAX_RETRIES)
                return None
            time.sleep(wait)

    return None


def build_collection_name(identifier: str) -> str:
    """Deterministic, short collection name from arbitrary text/bytes."""
    return f"col_{hashlib.sha256(identifier.encode()).hexdigest()[:12]}"
