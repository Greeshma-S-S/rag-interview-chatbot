"""
FastAPI backend for the AI Engineer RAG Chatbot.

Endpoints:
  POST /chat          — single-turn QA
  POST /chat/stream   — streaming QA (SSE)
  POST /interview     — mock interview mode
  GET  /health        — health check
  GET  /stats         — pipeline statistics
  POST /ingest        — trigger re-ingestion (admin)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import AsyncIterator, List, Optional

# Add parent to path so we can import config & app
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from config import settings
from app.rag_pipeline import RAGPipeline, RAGResponse

try:
    from app.logger import logger
except ImportError:
    from loguru import logger

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Engineer RAG Chatbot API",
    description="Production-ready RAG chatbot for AI/ML topics and mock interviews.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global RAG pipeline instance ──────────────────────────────────────────────
rag: RAGPipeline | None = None


# ── Security ──────────────────────────────────────────────────────────────────
security = HTTPBearer(auto_error=False)


def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Security(security)):
    """Simple bearer token auth for admin endpoints."""
    if credentials and credentials.credentials == settings.api_secret_key:
        return True
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API token",
    )


# ── Pydantic Models ───────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="User question")
    chat_history: List[Message] = Field(default=[], description="Prior conversation turns")
    top_k: Optional[int] = Field(default=None, ge=1, le=20, description="Number of chunks to retrieve")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is Retrieval-Augmented Generation and why is it useful?",
                "chat_history": [],
                "top_k": 5,
            }
        }


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    retrieval_scores: List[float]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    pipeline_ready: bool
    doc_count: int
    model: str


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    global rag
    logger.info("Starting AI Engineer RAG Chatbot API …")
    rag = RAGPipeline()
    ok = rag.initialise()
    if ok:
        logger.info(f"RAG pipeline ready — {rag.stats}")
    else:
        logger.warning(
            "RAG pipeline NOT ready. "
            "Run `python scripts/download_pdfs.py && python scripts/ingest_data.py`"
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_pipeline() -> RAGPipeline:
    if rag is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")
    return rag


def messages_to_history(messages: List[Message]) -> List[dict]:
    return [{"role": m.role, "content": m.content} for m in messages]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    pipeline = get_pipeline()
    stats = pipeline.stats
    return HealthResponse(
        status="healthy" if stats["is_ready"] else "degraded",
        pipeline_ready=stats["is_ready"],
        doc_count=stats["doc_count"],
        model=stats["llm_model"],
    )


@app.get("/stats", tags=["System"])
async def stats():
    pipeline = get_pipeline()
    return pipeline.stats


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(req: ChatRequest):
    """Single-turn QA with RAG."""
    pipeline = get_pipeline()
    start = time.time()

    try:
        response: RAGResponse = pipeline.query(
            question=req.question,
            chat_history=messages_to_history(req.chat_history),
            mode="qa",
            k=req.top_k,
        )
    except Exception as e:
        logger.exception(f"Error during /chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = (time.time() - start) * 1000

    return ChatResponse(
        answer=response.answer,
        sources=response.sources,
        retrieval_scores=response.retrieval_scores,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/chat/stream", tags=["Chat"])
async def chat_stream(req: ChatRequest):
    """Streaming QA with RAG — returns Server-Sent Events."""
    pipeline = get_pipeline()

    async def event_generator() -> AsyncIterator[str]:
        try:
            for token in pipeline.query_stream(
                question=req.question,
                chat_history=messages_to_history(req.chat_history),
                mode="qa",
                k=req.top_k,
            ):
                yield f"data: {token}\n\n"
        except Exception as e:
            logger.exception(f"Streaming error: {e}")
            yield f"data: [ERROR] {e}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/interview", response_model=ChatResponse, tags=["Interview"])
async def interview(req: ChatRequest):
    """Mock interview mode — structured Q&A with evaluation."""
    pipeline = get_pipeline()
    start = time.time()

    try:
        response: RAGResponse = pipeline.query(
            question=req.question,
            chat_history=messages_to_history(req.chat_history),
            mode="interview",
            k=req.top_k,
        )
    except Exception as e:
        logger.exception(f"Error during /interview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = (time.time() - start) * 1000

    return ChatResponse(
        answer=response.answer,
        sources=response.sources,
        retrieval_scores=response.retrieval_scores,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/ingest", tags=["Admin"])
async def trigger_ingest(_: bool = Depends(verify_token)):
    """
    Admin endpoint: trigger a rebuild of the vector store.
    Requires Bearer token authentication.
    """
    pipeline = get_pipeline()
    from scripts.ingest_data import main as ingest_main  # type: ignore
    import threading

    def run_ingest():
        pipeline.initialise(force_rebuild=True)

    t = threading.Thread(target=run_ingest, daemon=True)
    t.start()
    return {"status": "Ingestion started in background"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        workers=1,
        log_level=settings.log_level.lower(),
    )
