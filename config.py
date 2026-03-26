"""
Central configuration for the AI Engineer RAG Chatbot.
All settings are loaded from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


BASE_DIR = Path(__file__).parent


class Settings(BaseSettings):
    # ── OpenAI ────────────────────────────────────────────────
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")

    # ── Model ─────────────────────────────────────────────────
    llm_model: str = Field(default="gpt-4o", env="LLM_MODEL")
    embedding_model: str = Field(
        default="text-embedding-3-small", env="EMBEDDING_MODEL"
    )
    embedding_dimensions: int = Field(default=1536, env="EMBEDDING_DIMENSIONS")
    llm_temperature: float = Field(default=0.0, env="LLM_TEMPERATURE")
    max_tokens: int = Field(default=2048, env="MAX_TOKENS")

    # ── RAG ───────────────────────────────────────────────────
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    top_k_retrieval: int = Field(default=5, env="TOP_K_RETRIEVAL")
    vector_store_path: str = Field(
        default="data/vectorstore", env="VECTOR_STORE_PATH"
    )

    # ── Data ──────────────────────────────────────────────────
    pdf_data_path: str = Field(default="data/pdfs", env="PDF_DATA_PATH")

    # ── API ───────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_secret_key: str = Field(
        default="change-me-to-a-random-secret-key", env="API_SECRET_KEY"
    )

    # ── AWS ───────────────────────────────────────────────────
    aws_region: str = Field(default="us-east-1", env="AWS_REGION")
    aws_s3_bucket: str = Field(default="", env="AWS_S3_BUCKET")
    aws_access_key_id: str = Field(default="", env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(default="", env="AWS_SECRET_ACCESS_KEY")

    # ── Logging ───────────────────────────────────────────────
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def vector_store_dir(self) -> Path:
        path = BASE_DIR / self.vector_store_path
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def pdf_data_dir(self) -> Path:
        path = BASE_DIR / self.pdf_data_path
        path.mkdir(parents=True, exist_ok=True)
        return path


# Singleton instance – import this everywhere
settings = Settings()
