"""
RAGAS Evaluation Module.

Evaluates the RAG pipeline using the RAGAS framework with metrics:
  - Faithfulness          — is the answer grounded in the retrieved context?
  - Answer Relevancy      — does the answer address the question?
  - Context Precision     — are retrieved chunks relevant?
  - Context Recall        — does retrieval capture necessary information?
  - Answer Correctness    — factual correctness vs ground truth (needs reference)

Usage:
    python -m app.evaluation
    # or
    from app.evaluation import RAGEvaluator
    evaluator = RAGEvaluator(pipeline)
    results = evaluator.evaluate(test_dataset)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import settings

try:
    from app.logger import logger
except ImportError:
    from loguru import logger

try:
    from ragas import evaluate
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logger.warning(
        "RAGAS not available. Install with: pip install ragas datasets langchain-openai"
    )


# ── Sample test dataset for AI/ML topics ─────────────────────────────────────
SAMPLE_TEST_DATASET: List[Dict[str, Any]] = [
    {
        "question": "What is Retrieval-Augmented Generation (RAG)?",
        "ground_truth": (
            "RAG combines a retrieval system with a generative language model. "
            "It retrieves relevant documents from a knowledge base and uses them "
            "as context for the LLM to generate more accurate, grounded answers."
        ),
    },
    {
        "question": "What is the difference between LoRA and full fine-tuning?",
        "ground_truth": (
            "Full fine-tuning updates all model parameters while LoRA only trains "
            "small low-rank decomposition matrices added to attention layers, "
            "drastically reducing memory requirements and training time."
        ),
    },
    {
        "question": "What are the key components of the Transformer architecture?",
        "ground_truth": (
            "The Transformer consists of encoder and decoder stacks, each containing "
            "multi-head self-attention layers, feed-forward networks, layer normalisation, "
            "and residual connections. It uses positional encoding instead of recurrence."
        ),
    },
    {
        "question": "What is the ReAct pattern in AI agents?",
        "ground_truth": (
            "ReAct interleaves reasoning traces (Thought) with actions (Act) and "
            "observations (Observe) in a loop, allowing LLMs to reason about which "
            "tool to call, use it, observe the result, and continue reasoning."
        ),
    },
    {
        "question": "What metrics does RAGAS use to evaluate RAG systems?",
        "ground_truth": (
            "RAGAS evaluates: faithfulness (answer grounded in context), "
            "answer relevancy (answers the question), context precision "
            "(relevance of retrieved chunks), and context recall "
            "(coverage of necessary information)."
        ),
    },
    {
        "question": "What is chain-of-thought prompting?",
        "ground_truth": (
            "Chain-of-thought prompting encourages LLMs to generate intermediate "
            "reasoning steps before the final answer, significantly improving "
            "performance on complex reasoning, math, and multi-step tasks."
        ),
    },
    {
        "question": "How does FAISS perform similarity search?",
        "ground_truth": (
            "FAISS uses approximate nearest neighbor algorithms to search high-dimensional "
            "vector spaces efficiently. It builds index structures (like IVF, HNSW, or flat "
            "indexes) to find the most similar vectors to a query without scanning all vectors."
        ),
    },
    {
        "question": "What is QLoRA and how does it differ from LoRA?",
        "ground_truth": (
            "QLoRA combines LoRA with 4-bit quantisation of the base model weights. "
            "It uses NF4 quantisation, double quantisation, and paged optimisers to "
            "fine-tune large models (65B+) on a single GPU with minimal quality loss."
        ),
    },
]


@dataclass
class EvaluationResult:
    """Holds the results of a RAGAS evaluation run."""
    metrics: Dict[str, float] = field(default_factory=dict)
    num_samples: int = 0
    failed_samples: int = 0
    raw_results: Optional[Any] = None

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "RAGAS Evaluation Results",
            "=" * 50,
            f"Samples evaluated: {self.num_samples}",
            f"Failed:            {self.failed_samples}",
            "",
            "Metrics:",
        ]
        for metric, score in self.metrics.items():
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            lines.append(f"  {metric:25s} {score:.4f}  [{bar}]")
        lines.append("=" * 50)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "metrics": self.metrics,
            "num_samples": self.num_samples,
            "failed_samples": self.failed_samples,
        }


class RAGEvaluator:
    """
    Evaluates a RAGPipeline using RAGAS metrics.
    """

    def __init__(self, pipeline=None):
        self.pipeline = pipeline
        self._setup_ragas()

    def _setup_ragas(self):
        """Initialise RAGAS with OpenAI LLM and embeddings."""
        if not RAGAS_AVAILABLE:
            return

        llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=settings.llm_model,
                temperature=0,
                openai_api_key=settings.openai_api_key,
            )
        )
        emb = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model=settings.embedding_model,
                openai_api_key=settings.openai_api_key,
            )
        )

        # Configure metrics
        self.metrics = [faithfulness, answer_relevancy, context_precision]
        for m in self.metrics:
            m.llm = llm
            if hasattr(m, "embeddings"):
                m.embeddings = emb

        # Add recall & correctness only if ground truth is provided
        self.metrics_with_reference = [
            faithfulness, answer_relevancy,
            context_precision, context_recall,
            answer_correctness,
        ]
        for m in self.metrics_with_reference:
            m.llm = llm
            if hasattr(m, "embeddings"):
                m.embeddings = emb

        logger.info("RAGAS evaluator configured")

    # ── Public API ────────────────────────────────────────────

    def evaluate(
        self,
        test_cases: Optional[List[Dict]] = None,
        use_ground_truth: bool = True,
        save_path: Optional[Path] = None,
    ) -> EvaluationResult:
        """
        Run RAGAS evaluation.

        Args:
            test_cases: List of dicts with 'question' and optionally 'ground_truth'.
                        Defaults to SAMPLE_TEST_DATASET.
            use_ground_truth: Whether to include ground truth metrics.
            save_path: Optional path to save results JSON.

        Returns:
            EvaluationResult with per-metric scores.
        """
        if not RAGAS_AVAILABLE:
            logger.error("RAGAS is not installed. Cannot evaluate.")
            return EvaluationResult()

        if self.pipeline is None:
            raise ValueError("No RAG pipeline provided to evaluator.")

        cases = test_cases or SAMPLE_TEST_DATASET
        logger.info(f"Evaluating {len(cases)} test cases …")

        eval_data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }

        failed = 0
        for case in cases:
            try:
                q = case["question"]
                response = self.pipeline.query(q, mode="qa")

                eval_data["question"].append(q)
                eval_data["answer"].append(response.answer)
                eval_data["contexts"].append(
                    [doc.page_content for doc in response.source_documents]
                )
                eval_data["ground_truth"].append(
                    case.get("ground_truth", "")
                )
                logger.debug(f"  ✓ Evaluated: {q[:60]}…")
            except Exception as e:
                logger.warning(f"  ✗ Failed to evaluate '{case.get('question', '?')}': {e}")
                failed += 1

        if not eval_data["question"]:
            logger.error("No test cases evaluated successfully.")
            return EvaluationResult(failed_samples=failed)

        dataset = Dataset.from_dict(eval_data)
        metrics = (
            self.metrics_with_reference
            if use_ground_truth
            else self.metrics
        )

        logger.info("Running RAGAS evaluation …")
        raw = evaluate(dataset, metrics=metrics)

        scores = {}
        for metric_name in [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "answer_correctness",
        ]:
            if metric_name in raw:
                scores[metric_name] = float(raw[metric_name])

        result = EvaluationResult(
            metrics=scores,
            num_samples=len(eval_data["question"]),
            failed_samples=failed,
            raw_results=raw,
        )

        logger.info(f"\n{result.summary()}")

        if save_path:
            save_path.write_text(json.dumps(result.to_dict(), indent=2))
            logger.info(f"Results saved to {save_path}")

        return result


# ── Standalone runner ─────────────────────────────────────────────────────────

def run_evaluation():
    """Entry point for standalone evaluation."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from app.rag_pipeline import RAGPipeline

    logger.info("Initialising RAG pipeline for evaluation …")
    pipeline = RAGPipeline()
    if not pipeline.initialise():
        logger.error("Pipeline not ready — run ingest_data.py first.")
        sys.exit(1)

    evaluator = RAGEvaluator(pipeline=pipeline)
    results = evaluator.evaluate(
        save_path=Path("data/evaluation_results.json")
    )
    print(results.summary())


if __name__ == "__main__":
    run_evaluation()
