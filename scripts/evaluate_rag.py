#!/usr/bin/env python3
"""
Run RAGAS evaluation on the RAG pipeline.

Usage:
    python scripts/evaluate_rag.py
    python scripts/evaluate_rag.py --samples 5     # evaluate on 5 samples
    python scripts/evaluate_rag.py --no-ground-truth
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag_pipeline import RAGPipeline
from app.evaluation import RAGEvaluator, SAMPLE_TEST_DATASET

try:
    from app.logger import logger
except ImportError:
    from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument("--samples", type=int, default=None, help="Number of test cases to run")
    parser.add_argument("--no-ground-truth", action="store_true", help="Evaluate without reference answers")
    parser.add_argument("--output", type=str, default="data/evaluation_results.json", help="Output JSON path")
    args = parser.parse_args()

    # Initialise pipeline
    pipeline = RAGPipeline()
    if not pipeline.initialise():
        logger.error("Pipeline is not ready. Run ingest_data.py first.")
        sys.exit(1)

    # Prepare test cases
    test_cases = SAMPLE_TEST_DATASET
    if args.samples:
        test_cases = test_cases[: args.samples]

    logger.info(f"Evaluating with {len(test_cases)} test cases …")

    # Run evaluation
    evaluator = RAGEvaluator(pipeline=pipeline)
    results = evaluator.evaluate(
        test_cases=test_cases,
        use_ground_truth=not args.no_ground_truth,
        save_path=Path(args.output),
    )

    print(results.summary())


if __name__ == "__main__":
    main()
