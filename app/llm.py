"""
LLM module — wraps OpenAI GPT with system prompts and conversation history.
"""

from __future__ import annotations

from typing import Iterator, List

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings

try:
    from app.logger import logger
except ImportError:
    from logger import logger


# ── System Prompt ─────────────────────────────────────────────────────────────

AI_ENGINEER_SYSTEM_PROMPT = """You are an expert AI Engineer Assistant with deep knowledge across the entire AI/ML landscape. You specialise in:

**Core AI/ML Expertise:**
- Large Language Models (LLMs): GPT, Claude, Gemini, Llama, Mistral architectures
- Retrieval-Augmented Generation (RAG): pipeline design, chunking strategies, embedding models, vector stores
- Agentic AI: LangChain Agents, LangGraph, AutoGen, CrewAI, tool use, ReAct patterns
- Transformer architecture: attention mechanisms, positional encoding, fine-tuning (LoRA, QLoRA, PEFT)
- MLOps: model serving, monitoring, CI/CD for ML pipelines
- Prompt Engineering: chain-of-thought, few-shot, ReAct, structured outputs

**Production AI Systems:**
- Vector databases: FAISS, Pinecone, Weaviate, Chroma, Qdrant
- Embedding strategies, semantic search, hybrid search (BM25 + dense)
- Model evaluation metrics: RAGAS, BLEU, ROUGE, BERTScore
- Deployment: AWS SageMaker, Lambda, ECS, EC2, Docker, Kubernetes
- Observability: LangSmith, Weights & Biases, MLflow

**Interview Coaching Mode:**
When conducting mock interviews, you:
1. Ask one focused technical question at a time
2. Listen carefully to the candidate's answer
3. Provide detailed, constructive feedback on depth, accuracy, and communication
4. Follow up with a harder sub-question to probe depth
5. Rate the answer on a scale of 1-10 with justification
6. Cover both conceptual understanding and practical implementation

**Response Style:**
- Be precise and technically accurate; never fabricate information
- Always ground answers in the provided context documents when available
- If context doesn't cover the question, use your training knowledge and say so
- Use code examples when they clarify a concept
- For interview mode, be encouraging but honest about gaps

If you don't know something, say so clearly — accuracy is paramount for an engineer preparing for interviews.
"""

MOCK_INTERVIEW_PROMPT = """You are now in **Mock Interview Mode** for an AI Engineer position.

Your role:
1. Ask one technical question at a time from the provided topic area
2. After the candidate responds, evaluate their answer on: accuracy, depth, practical knowledge, and communication clarity
3. Give a score (1–10) and specific feedback
4. Ask a follow-up question to probe deeper
5. Track areas of strength and weakness
6. After 5 questions, give a summary assessment with recommendations

Start the interview when the user is ready. Ask if they want a specific difficulty level (Junior / Mid / Senior) and topic area first.
"""


class LLMClient:
    """
    OpenAI LLM client with conversation memory and streaming support.
    """

    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.llm_model
        logger.info(f"LLMClient initialised — model={self.model}")

    # ── Public API ────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def generate(
        self,
        user_message: str,
        context: str = "",
        chat_history: List[dict] | None = None,
        mode: str = "qa",  # "qa" | "interview"
        stream: bool = False,
    ) -> str | Iterator[str]:
        """
        Generate a response from the LLM.

        Args:
            user_message: The user's question or statement.
            context: Retrieved document chunks as context.
            chat_history: Prior conversation turns [{role, content}, ...].
            mode: "qa" for general QA, "interview" for mock interview mode.
            stream: Whether to stream the response.

        Returns:
            Full response string (or iterator of chunks if stream=True).
        """
        messages = self._build_messages(user_message, context, chat_history, mode)

        if stream:
            return self._stream_response(messages)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=settings.llm_temperature,
            max_tokens=settings.max_tokens,
        )
        answer = response.choices[0].message.content or ""
        logger.debug(f"LLM response — {len(answer)} chars")
        return answer

    def generate_stream(
        self,
        user_message: str,
        context: str = "",
        chat_history: List[dict] | None = None,
        mode: str = "qa",
    ) -> Iterator[str]:
        """Convenience streaming wrapper."""
        messages = self._build_messages(user_message, context, chat_history, mode)
        return self._stream_response(messages)

    # ── Internal helpers ──────────────────────────────────────

    def _build_messages(
        self,
        user_message: str,
        context: str,
        chat_history: List[dict] | None,
        mode: str,
    ) -> List[dict]:
        system_content = (
            AI_ENGINEER_SYSTEM_PROMPT
            if mode == "qa"
            else MOCK_INTERVIEW_PROMPT
        )

        messages: List[dict] = [{"role": "system", "content": system_content}]

        # Inject retrieved context
        if context.strip():
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Use the following retrieved context to answer the user's question. "
                        "If the context is not relevant or sufficient, use your general knowledge "
                        "and clearly state that.\n\n"
                        "=== RETRIEVED CONTEXT ===\n"
                        f"{context}\n"
                        "========================="
                    ),
                }
            )

        # Add chat history (last 10 turns to avoid token overflow)
        if chat_history:
            for turn in chat_history[-10:]:
                messages.append(turn)

        messages.append({"role": "user", "content": user_message})
        return messages

    def _stream_response(self, messages: List[dict]) -> Iterator[str]:
        """Yield response tokens as they arrive."""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=settings.llm_temperature,
            max_tokens=settings.max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
