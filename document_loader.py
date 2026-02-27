"""
Document Q&A Assistant — Document Loading & Chunking
Handles PDF, DOCX, TXT, MD, CSV and Wikipedia ingestion.
"""

import os
import re
import hashlib
import tempfile
from datetime import datetime
from typing import List, Optional

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    WikipediaLoader,
    CSVLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from logger import logger
from config import SUPPORTED_EXTENSIONS


# ── Text pre-processing ────────────────────────────────────
def preprocess_text(text: str) -> str:
    """Remove control chars and collapse whitespace."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_file_hash(content: bytes) -> str:
    """SHA-256 hash of raw file bytes (more collision-resistant than MD5)."""
    return hashlib.sha256(content).hexdigest()


# ── Loaders ─────────────────────────────────────────────────
_LOADER_MAP = {
    ".pdf":  lambda p: PyPDFLoader(p),
    ".docx": lambda p: Docx2txtLoader(p),
    ".txt":  lambda p: TextLoader(p, encoding="utf-8"),
    ".md":   lambda p: TextLoader(p, encoding="utf-8"),
    ".csv":  lambda p: CSVLoader(p, encoding="utf-8"),
}


def load_uploaded_file(file) -> Optional[List[Document]]:
    """
    Load a Streamlit UploadedFile into LangChain Documents.
    Returns None on unsupported format or error.
    """
    _, ext = os.path.splitext(file.name)
    ext = ext.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        logger.warning("Unsupported file extension: %s", ext)
        return None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        loader = _LOADER_MAP[ext](tmp_path)
        documents = loader.load()

        for doc in documents:
            doc.page_content = preprocess_text(doc.page_content)
            doc.metadata.update(
                {
                    "source": file.name,
                    "file_type": ext,
                    "file_size_kb": round(file.size / 1024, 2),
                    "upload_time": datetime.now().isoformat(),
                }
            )

        logger.info(
            "Loaded %d page(s) from '%s' (%s)",
            len(documents),
            file.name,
            ext,
        )
        return documents

    except Exception as exc:
        logger.exception("Failed to load '%s': %s", file.name, exc)
        return None

    finally:
        # Always clean up temp file
        if "tmp_path" in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def load_wikipedia(query: str, max_docs: int = 2) -> Optional[List[Document]]:
    """Fetch Wikipedia articles for *query*."""
    try:
        loader = WikipediaLoader(query=query, load_max_docs=max_docs)
        documents = loader.load()
        for doc in documents:
            doc.page_content = preprocess_text(doc.page_content)
        logger.info("Wikipedia: loaded %d article(s) for '%s'", len(documents), query)
        return documents
    except Exception as exc:
        logger.exception("Wikipedia load failed for '%s': %s", query, exc)
        return None


# ── Chunking ────────────────────────────────────────────────
def chunk_documents(
    documents: List[Document],
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> List[Document]:
    """Split documents into overlapping chunks and tag metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["chunk_total"] = len(chunks)

    logger.info("Chunked %d document(s) → %d chunks", len(documents), len(chunks))
    return chunks
