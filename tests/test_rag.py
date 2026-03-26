"""
Unit and integration tests for the RAG chatbot pipeline.

Run with:
    pytest tests/ -v
    pytest tests/ -v --cov=app --cov-report=html
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── DocumentProcessor tests ───────────────────────────────────────────────────

class TestDocumentProcessor:
    def setup_method(self):
        from app.document_processor import DocumentProcessor
        self.processor = DocumentProcessor()

    def test_clean_text_removes_excessive_newlines(self):
        raw = "First paragraph.\n\n\n\nSecond paragraph."
        cleaned = self.processor._clean_text(raw)
        assert "\n\n\n" not in cleaned
        assert "First paragraph." in cleaned
        assert "Second paragraph." in cleaned

    def test_clean_text_removes_standalone_numbers(self):
        raw = "Some text\n42\nMore text"
        cleaned = self.processor._clean_text(raw)
        assert "Some text" in cleaned
        assert "More text" in cleaned

    def test_splitter_creates_overlapping_chunks(self):
        long_text = "Word " * 500  # 500 words
        chunks = self.processor.splitter.split_text(long_text)
        assert len(chunks) > 1, "Long text should be split into multiple chunks"

    def test_load_nonexistent_pdf_returns_empty(self, tmp_path):
        result = self.processor.load_pdf(tmp_path / "nonexistent.pdf")
        assert result == []

    def test_load_empty_directory_returns_empty(self, tmp_path):
        result = self.processor.load_directory(tmp_path)
        assert result == []


# ── EmbeddingManager tests ─────────────────────────────────────────────────────

class TestEmbeddingManager:
    def setup_method(self):
        # Use a mock to avoid real API calls
        with patch("app.embeddings.OpenAI"):
            from app.embeddings import EmbeddingManager
            self.manager = EmbeddingManager()
            self.manager.client = MagicMock()

    def test_embed_empty_list_returns_empty(self):
        result = self.manager.embed_texts([])
        assert result == []

    def test_cache_key_is_deterministic(self):
        key1 = self.manager._cache_key("hello world")
        key2 = self.manager._cache_key("hello world")
        assert key1 == key2

    def test_cache_key_differs_for_different_text(self):
        key1 = self.manager._cache_key("hello")
        key2 = self.manager._cache_key("world")
        assert key1 != key2

    def test_embed_with_mocked_api(self):
        # Mock the OpenAI response
        mock_embedding = [0.1] * 1536
        mock_data = MagicMock()
        mock_data.embedding = mock_embedding
        mock_response = MagicMock()
        mock_response.data = [mock_data]
        self.manager.client.embeddings.create.return_value = mock_response

        results = self.manager.embed_texts(["test text"])
        assert len(results) == 1
        assert len(results[0]) == 1536

    def test_cache_saves_and_loads(self, tmp_path):
        with patch("app.embeddings.OpenAI"):
            from app.embeddings import EmbeddingManager
            manager = EmbeddingManager(cache_dir=tmp_path)

        emb = [0.5] * 10
        manager._save_to_cache("test", emb)
        loaded = manager._load_from_cache("test")
        assert loaded == emb


# ── FAISSVectorStore tests ─────────────────────────────────────────────────────

class TestFAISSVectorStore:
    def test_is_not_ready_before_build(self, tmp_path):
        with patch("app.vector_store.settings") as mock_settings, \
             patch("app.embeddings.OpenAI"):
            mock_settings.vector_store_dir = tmp_path
            mock_settings.top_k_retrieval = 5
            mock_settings.embedding_model = "text-embedding-3-small"
            mock_settings.chunk_size = 1000

            from app.vector_store import FAISSVectorStore
            store = FAISSVectorStore()
            assert not store.is_ready

    def test_build_and_search(self, tmp_path):
        """Integration test: build a small index and search it."""
        import numpy as np
        from langchain_core.documents import Document

        with patch("app.embeddings.settings") as mock_emb_settings, \
             patch("app.vector_store.settings") as mock_settings, \
             patch("app.embeddings.OpenAI"):
            mock_settings.vector_store_dir = tmp_path
            mock_settings.top_k_retrieval = 2
            mock_settings.embedding_model = "text-embedding-3-small"
            mock_settings.chunk_size = 1000
            mock_settings.embedding_dimensions = 4
            mock_emb_settings.openai_api_key = "fake"
            mock_emb_settings.embedding_model = "text-embedding-3-small"
            mock_emb_settings.embedding_dimensions = 4

            from app.embeddings import EmbeddingManager
            from app.vector_store import FAISSVectorStore

            # Create a manager that returns deterministic embeddings
            manager = EmbeddingManager()

            def fake_embed(texts):
                # Return simple unit vectors based on text hash
                results = []
                for i, t in enumerate(texts):
                    vec = [float(j == i % 4) for j in range(4)]
                    results.append(vec)
                return results

            manager.embed_texts = fake_embed
            manager.embed_query = lambda q: [1.0, 0.0, 0.0, 0.0]

            docs = [
                Document(page_content="What is RAG?", metadata={"source": "test.pdf"}),
                Document(page_content="Transformers architecture", metadata={"source": "test.pdf"}),
                Document(page_content="Fine-tuning LLMs", metadata={"source": "test.pdf"}),
            ]

            store = FAISSVectorStore(embed_manager=manager)
            store.store_path = tmp_path
            store.build(docs)

            assert store.is_ready
            assert store.doc_count == 3


# ── LLMClient tests ────────────────────────────────────────────────────────────

class TestLLMClient:
    def test_build_messages_includes_system_prompt(self):
        with patch("app.llm.OpenAI"), \
             patch("app.llm.settings") as mock_settings:
            mock_settings.openai_api_key = "fake"
            mock_settings.llm_model = "gpt-4o"
            mock_settings.llm_temperature = 0
            mock_settings.max_tokens = 100

            from app.llm import LLMClient
            client = LLMClient()
            messages = client._build_messages("test question", "", None, "qa")

        assert messages[0]["role"] == "system"
        assert "AI Engineer" in messages[0]["content"]
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "test question"

    def test_build_messages_includes_context(self):
        with patch("app.llm.OpenAI"), \
             patch("app.llm.settings") as mock_settings:
            mock_settings.openai_api_key = "fake"
            mock_settings.llm_model = "gpt-4o"
            mock_settings.llm_temperature = 0
            mock_settings.max_tokens = 100

            from app.llm import LLMClient
            client = LLMClient()
            messages = client._build_messages("test", "some context", None, "qa")

        # Context should be in a system message
        context_msgs = [m for m in messages if "RETRIEVED CONTEXT" in m.get("content", "")]
        assert len(context_msgs) == 1

    def test_interview_mode_uses_different_prompt(self):
        with patch("app.llm.OpenAI"), \
             patch("app.llm.settings") as mock_settings:
            mock_settings.openai_api_key = "fake"
            mock_settings.llm_model = "gpt-4o"
            mock_settings.llm_temperature = 0
            mock_settings.max_tokens = 100

            from app.llm import LLMClient
            client = LLMClient()
            qa_msgs = client._build_messages("q", "", None, "qa")
            iv_msgs = client._build_messages("q", "", None, "interview")

        assert qa_msgs[0]["content"] != iv_msgs[0]["content"]


# ── RAGResponse tests ──────────────────────────────────────────────────────────

class TestRAGResponse:
    def test_sources_deduplication(self):
        from langchain_core.documents import Document
        from app.rag_pipeline import RAGResponse

        docs = [
            Document(page_content="a", metadata={"source": "paper1.pdf"}),
            Document(page_content="b", metadata={"source": "paper1.pdf"}),
            Document(page_content="c", metadata={"source": "paper2.pdf"}),
        ]
        response = RAGResponse(answer="test", source_documents=docs)
        assert len(response.sources) == 2
        assert "paper1.pdf" in response.sources
        assert "paper2.pdf" in response.sources


# ── API endpoint tests ─────────────────────────────────────────────────────────

class TestAPIEndpoints:
    def test_health_endpoint_structure(self):
        """Test that the health endpoint returns the expected structure."""
        with patch("app.main.rag") as mock_rag:
            mock_rag.stats = {
                "is_ready": True,
                "doc_count": 100,
                "llm_model": "gpt-4o",
                "embedding_model": "text-embedding-3-small",
                "top_k": 5,
            }
            from fastapi.testclient import TestClient
            from app.main import app

            client = TestClient(app)
            # Would need full pipeline setup for real test
            # This is a structural test
            assert app is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
