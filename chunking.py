"""
Document Q&A Assistant — Smart Chunking Strategies
Provides semantic chunking (embedding-based boundary detection) with fallback
to recursive character splitting.
"""

from typing import Dict, List, Optional
import numpy as np

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import EMBEDDING_MODEL, EMBEDDING_DIMENSIONS
from logger import logger


# ─────────────────────────────────────────────────────────────
# 1. Semantic Chunker — splits at *meaning* boundaries
# ─────────────────────────────────────────────────────────────
class SemanticChunker:
    """
    Splits text by detecting semantic shift between sentences.
    Algorithm:
      1. Split into sentences
      2. Embed every sentence
      3. Compute cosine similarity between consecutive sentence embeddings
      4. Split where similarity drops below a dynamic threshold
         (mean - breakpoint_percentile_threshold * std)
      5. Group sentences between breakpoints into chunks
    This keeps semantically coherent paragraphs together instead of
    blindly cutting at a character count.
    """

    def __init__(
        self,
        breakpoint_percentile: float = 80.0,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
    ):
        self.breakpoint_percentile = breakpoint_percentile
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self._embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIMENSIONS,
        )

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Rough sentence splitter that handles common abbreviations."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        a_arr, b_arr = np.array(a), np.array(b)
        denom = (np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
        if denom == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / denom)

    def _find_breakpoints(self, similarities: List[float]) -> List[int]:
        """Return indices where semantic similarity drops below threshold."""
        if not similarities:
            return []
        arr = np.array(similarities)
        threshold = np.percentile(arr, 100 - self.breakpoint_percentile)
        return [i for i, s in enumerate(similarities) if s < threshold]

    def split_text(self, text: str, metadata: Optional[dict] = None) -> List[Document]:
        """Split *text* into semantically coherent chunks."""
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [Document(page_content=text, metadata=metadata or {})]

        # Embed all sentences in one batch
        try:
            embeddings = self._embeddings.embed_documents(sentences)
        except Exception as exc:
            logger.warning("Semantic embedding failed (%s), falling back to recursive.", exc)
            return self._fallback_split(text, metadata)

        # Cosine similarities between consecutive sentences
        similarities = [
            self._cosine_similarity(embeddings[i], embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ]

        # Find breakpoints
        breakpoints = self._find_breakpoints(similarities)

        # Group sentences into chunks
        chunks: List[Document] = []
        start = 0
        for bp in breakpoints:
            chunk_text = " ".join(sentences[start : bp + 1]).strip()
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(Document(
                    page_content=chunk_text,
                    metadata={**(metadata or {}), "chunking_method": "semantic"},
                ))
            start = bp + 1

        # Remaining sentences
        remaining = " ".join(sentences[start:]).strip()
        if remaining:
            if chunks and len(remaining) < self.min_chunk_size:
                # Merge tiny tail into last chunk
                chunks[-1].page_content += " " + remaining
            else:
                chunks.append(Document(
                    page_content=remaining,
                    metadata={**(metadata or {}), "chunking_method": "semantic"},
                ))

        # Split any oversized chunks with recursive splitter
        final_chunks = []
        for chunk in chunks:
            if len(chunk.page_content) > self.max_chunk_size:
                sub = self._recursive_split(chunk.page_content, chunk.metadata)
                final_chunks.extend(sub)
            else:
                final_chunks.append(chunk)

        return final_chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split a list of Documents semantically."""
        all_chunks = []
        for doc in documents:
            chunks = self.split_text(doc.page_content, doc.metadata)
            all_chunks.extend(chunks)
        # Tag indices
        for i, c in enumerate(all_chunks):
            c.metadata["chunk_index"] = i
            c.metadata["chunk_total"] = len(all_chunks)
        logger.info("Semantic chunking: %d docs → %d chunks", len(documents), len(all_chunks))
        return all_chunks

    def _fallback_split(self, text: str, metadata: Optional[dict] = None) -> List[Document]:
        return self._recursive_split(text, metadata)

    @staticmethod
    def _recursive_split(text: str, metadata: Optional[dict] = None) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700, chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        docs = splitter.create_documents([text], metadatas=[metadata or {}])
        for d in docs:
            d.metadata["chunking_method"] = "recursive_fallback"
        return docs


# ─────────────────────────────────────────────────────────────
# 2. Parent-Child Chunker — small chunks for retrieval,
#    return the larger parent for LLM context
# ─────────────────────────────────────────────────────────────
class ParentChildChunker:
    """
    Creates two tiers:
      • Parent chunks (large, ~1500 chars) — fed to GEN for full context
      • Child chunks  (small, ~300 chars)  — indexed for precise retrieval
    At query time, retrieve child → look up parent → send parent to LLM.
    """

    def __init__(
        self,
        parent_chunk_size: int = 1500,
        parent_overlap: int = 200,
        child_chunk_size: int = 300,
        child_overlap: int = 50,
    ):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def split_documents(self, documents: List[Document]):
        """
        Returns (child_chunks, parent_map) where parent_map maps
        parent_id → parent Document for context expansion.
        """
        parents = self.parent_splitter.split_documents(documents)
        parent_map = {}
        all_children = []

        for pid, parent in enumerate(parents):
            parent_id = f"parent_{pid}"
            parent.metadata["parent_id"] = parent_id
            parent_map[parent_id] = parent

            children = self.child_splitter.create_documents(
                [parent.page_content],
                metadatas=[{**parent.metadata, "parent_id": parent_id}],
            )
            for ci, child in enumerate(children):
                child.metadata["child_index"] = ci
                child.metadata["chunk_index"] = len(all_children)
            all_children.extend(children)

        # total
        for c in all_children:
            c.metadata["chunk_total"] = len(all_children)

        logger.info(
            "Parent-child chunking: %d docs → %d parents, %d children",
            len(documents), len(parents), len(all_children),
        )
        return all_children, parent_map


# ─────────────────────────────────────────────────────────────
# 3. Classic recursive (unchanged, kept as baseline)
# ─────────────────────────────────────────────────────────────
def chunk_recursive(
    documents: List[Document],
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> List[Document]:
    """Standard RecursiveCharacterTextSplitter chunking."""
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
        chunk.metadata["chunking_method"] = "recursive"
    logger.info("Recursive chunking: %d docs → %d chunks", len(documents), len(chunks))
    return chunks
