"""
Embedding management module.
Wraps OpenAI embeddings with caching and retry logic.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import List

import numpy as np
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings

try:
    from app.logger import logger
except ImportError:
    from logger import logger


class EmbeddingManager:
    """
    Manages text embeddings using OpenAI's embedding API.
    Includes optional disk-based caching to avoid redundant API calls.
    """

    def __init__(self, cache_dir: Path | None = None):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.embedding_model
        self.dimensions = settings.embedding_dimensions
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"EmbeddingManager initialised — model={self.model}")

    # ── Public API ────────────────────────────────────────────

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Return embeddings for a list of texts (batched, with cache)."""
        if not texts:
            return []

        results: List[List[float] | None] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        # Check cache first
        for i, text in enumerate(texts):
            cached = self._load_from_cache(text)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Batch embed uncached texts
        if uncached_texts:
            logger.debug(f"Embedding {len(uncached_texts)} uncached texts …")
            embeddings = self._embed_batch(uncached_texts)
            for idx, emb in zip(uncached_indices, embeddings):
                results[idx] = emb
                self._save_to_cache(texts[idx], emb)

        return results  # type: ignore[return-value]

    def embed_query(self, query: str) -> List[float]:
        """Return embedding for a single query string."""
        return self.embed_texts([query])[0]

    # ── Internal helpers ──────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Call OpenAI embeddings API with retry logic."""
        # OpenAI recommends batches ≤ 2048 inputs
        all_embeddings: List[List[float]] = []
        batch_size = 500

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
                dimensions=self.dimensions
                if "text-embedding-3" in self.model
                else None,
            )
            all_embeddings.extend([item.embedding for item in response.data])
            if i + batch_size < len(texts):
                time.sleep(0.1)  # gentle rate limit

        return all_embeddings

    def _cache_key(self, text: str) -> str:
        return hashlib.md5(f"{self.model}::{text}".encode()).hexdigest()

    def _cache_path(self, key: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{key}.json"

    def _load_from_cache(self, text: str) -> List[float] | None:
        path = self._cache_path(self._cache_key(text))
        if path and path.exists():
            return json.loads(path.read_text())
        return None

    def _save_to_cache(self, text: str, embedding: List[float]) -> None:
        path = self._cache_path(self._cache_key(text))
        if path:
            path.write_text(json.dumps(embedding))
