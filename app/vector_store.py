"""
Vector store module.
Builds and manages a FAISS index with OpenAI embeddings.
Also supports ChromaDB as an alternative backend.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import faiss
from langchain_core.documents import Document

from config import settings
from app.embeddings import EmbeddingManager

try:
    from app.logger import logger
except ImportError:
    from logger import logger


class FAISSVectorStore:
    """
    Production-grade FAISS vector store with persistence and similarity search.
    """

    INDEX_FILE = "faiss.index"
    DOCS_FILE = "documents.pkl"
    META_FILE = "metadata.json"

    def __init__(self, embed_manager: EmbeddingManager | None = None):
        self.embed_manager = embed_manager or EmbeddingManager()
        self.store_path = settings.vector_store_dir
        self.index: faiss.IndexFlatIP | None = None  # Inner Product (cosine after normalise)
        self.documents: List[Document] = []
        self._is_loaded = False

    # ── Build & Persist ───────────────────────────────────────

    def build(self, documents: List[Document], force_rebuild: bool = False) -> None:
        """Embed documents and build the FAISS index."""
        if self._is_loaded and not force_rebuild:
            logger.info("Vector store already loaded — skipping rebuild.")
            return

        logger.info(f"Building FAISS index for {len(documents)} documents …")
        texts = [doc.page_content for doc in documents]
        embeddings = self.embed_manager.embed_texts(texts)

        dim = len(embeddings[0])
        # Use IndexFlatIP (cosine similarity after normalisation)
        self.index = faiss.IndexFlatIP(dim)

        vectors = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(vectors)  # normalise for cosine similarity
        self.index.add(vectors)

        self.documents = documents
        self._is_loaded = True

        self._save()
        logger.info(f"FAISS index built — {self.index.ntotal} vectors, dim={dim}")

    def _save(self) -> None:
        """Persist index and documents to disk."""
        faiss.write_index(self.index, str(self.store_path / self.INDEX_FILE))
        with open(self.store_path / self.DOCS_FILE, "wb") as f:
            pickle.dump(self.documents, f)
        meta = {
            "num_docs": len(self.documents),
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
        }
        (self.store_path / self.META_FILE).write_text(json.dumps(meta, indent=2))
        logger.info(f"Vector store saved to {self.store_path}")

    def load(self) -> bool:
        """Load persisted index from disk. Returns True if successful."""
        idx_path = self.store_path / self.INDEX_FILE
        doc_path = self.store_path / self.DOCS_FILE

        if not (idx_path.exists() and doc_path.exists()):
            logger.warning("No persisted vector store found.")
            return False

        self.index = faiss.read_index(str(idx_path))
        with open(doc_path, "rb") as f:
            self.documents = pickle.load(f)
        self._is_loaded = True
        logger.info(
            f"Loaded FAISS index — {self.index.ntotal} vectors, "
            f"{len(self.documents)} documents"
        )
        return True

    # ── Search ────────────────────────────────────────────────

    def similarity_search(
        self, query: str, k: int | None = None
    ) -> List[Document]:
        """Return top-k most similar documents for a query."""
        docs_with_scores = self.similarity_search_with_scores(query, k=k)
        return [doc for doc, _ in docs_with_scores]

    def similarity_search_with_scores(
        self, query: str, k: int | None = None
    ) -> List[Tuple[Document, float]]:
        """Return top-k (document, score) pairs for a query."""
        if not self._is_loaded or self.index is None:
            raise RuntimeError("Vector store is not loaded. Call load() or build() first.")

        k = k or settings.top_k_retrieval
        query_vec = np.array(
            [self.embed_manager.embed_query(query)], dtype="float32"
        )
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            doc = self.documents[idx]
            doc.metadata["similarity_score"] = float(score)
            results.append((doc, float(score)))

        return results

    @property
    def is_ready(self) -> bool:
        return self._is_loaded and self.index is not None and self.index.ntotal > 0

    @property
    def doc_count(self) -> int:
        return len(self.documents) if self.documents else 0
