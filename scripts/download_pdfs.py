#!/usr/bin/env python3
"""
Download curated GenAI & Agentic AI papers from ArXiv and other open sources.
Papers cover: RAG, Transformers, Agents, LLMs, Prompt Engineering, and more.

Usage:
    python scripts/download_pdfs.py
    python scripts/download_pdfs.py --max 10   # download only 10 papers
"""

import argparse
import sys
import time
from pathlib import Path

import requests

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings

try:
    from app.logger import logger
except ImportError:
    from loguru import logger

# ── Curated paper list ────────────────────────────────────────────────────────
# Format: (filename, url, description)
PAPERS = [
    # ── Foundational Transformers ─────────────────────────────────────────────
    (
        "attention_is_all_you_need.pdf",
        "https://arxiv.org/pdf/1706.03762",
        "Attention Is All You Need — the original Transformer paper",
    ),
    (
        "bert_pretraining_nlp.pdf",
        "https://arxiv.org/pdf/1810.04805",
        "BERT: Pre-training of Deep Bidirectional Transformers",
    ),

    # ── Large Language Models ─────────────────────────────────────────────────
    (
        "gpt3_language_models_few_shot.pdf",
        "https://arxiv.org/pdf/2005.14165",
        "GPT-3: Language Models are Few-Shot Learners",
    ),
    (
        "llama2_open_foundation.pdf",
        "https://arxiv.org/pdf/2307.09288",
        "Llama 2: Open Foundation and Fine-Tuned Chat Models",
    ),
    (
        "instruction_tuning_flan.pdf",
        "https://arxiv.org/pdf/2109.01652",
        "Finetuned Language Models are Zero-Shot Learners",
    ),

    # ── RAG ───────────────────────────────────────────────────────────────────
    (
        "rag_knowledge_intensive_nlp.pdf",
        "https://arxiv.org/pdf/2005.11401",
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    ),
    (
        "self_rag.pdf",
        "https://arxiv.org/pdf/2310.11511",
        "Self-RAG: Learning to Retrieve, Generate, and Critique",
    ),
    (
        "corrective_rag.pdf",
        "https://arxiv.org/pdf/2401.15884",
        "Corrective Retrieval Augmented Generation (CRAG)",
    ),
    (
        "advanced_rag_survey.pdf",
        "https://arxiv.org/pdf/2312.10997",
        "Retrieval-Augmented Generation for Large Language Models: A Survey",
    ),

    # ── Agentic AI ────────────────────────────────────────────────────────────
    (
        "react_synergizing_reasoning_acting.pdf",
        "https://arxiv.org/pdf/2210.03629",
        "ReAct: Synergizing Reasoning and Acting in Language Models",
    ),
    (
        "toolformer_language_models_tools.pdf",
        "https://arxiv.org/pdf/2302.04761",
        "Toolformer: Language Models Can Teach Themselves to Use Tools",
    ),
    (
        "autogpt_autonomous_agents.pdf",
        "https://arxiv.org/pdf/2306.02224",
        "Auto-GPT for Online Decision Making",
    ),
    (
        "generative_agents.pdf",
        "https://arxiv.org/pdf/2304.03442",
        "Generative Agents: Interactive Simulacra of Human Behavior",
    ),
    (
        "agent_survey_llm_based.pdf",
        "https://arxiv.org/pdf/2308.11432",
        "A Survey on Large Language Model based Autonomous Agents",
    ),

    # ── Prompt Engineering ────────────────────────────────────────────────────
    (
        "chain_of_thought_prompting.pdf",
        "https://arxiv.org/pdf/2201.11903",
        "Chain-of-Thought Prompting Elicits Reasoning in LLMs",
    ),
    (
        "tree_of_thoughts.pdf",
        "https://arxiv.org/pdf/2305.10601",
        "Tree of Thoughts: Deliberate Problem Solving with LLMs",
    ),

    # ── Fine-tuning ───────────────────────────────────────────────────────────
    (
        "lora_low_rank_adaptation.pdf",
        "https://arxiv.org/pdf/2106.09685",
        "LoRA: Low-Rank Adaptation of Large Language Models",
    ),
    (
        "qlora_efficient_finetuning.pdf",
        "https://arxiv.org/pdf/2305.14314",
        "QLoRA: Efficient Finetuning of Quantized LLMs",
    ),

    # ── Evaluation ────────────────────────────────────────────────────────────
    (
        "ragas_evaluation_rag.pdf",
        "https://arxiv.org/pdf/2309.15217",
        "RAGAS: Automated Evaluation of Retrieval Augmented Generation",
    ),

    # ── Multimodal / GPT-4 ────────────────────────────────────────────────────
    (
        "gpt4_technical_report.pdf",
        "https://arxiv.org/pdf/2303.08774",
        "GPT-4 Technical Report",
    ),
]


def download_pdf(url: str, dest_path: Path, retries: int = 3) -> bool:
    """Download a PDF from a URL with retry logic."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; RAG-Chatbot-Downloader/1.0; "
            "+https://github.com/ai-engineer-chatbot)"
        )
    }

    for attempt in range(1, retries + 1):
        try:
            logger.info(f"  Downloading {dest_path.name} (attempt {attempt}) …")
            response = requests.get(url, headers=headers, timeout=60, stream=True)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if "pdf" not in content_type and "octet-stream" not in content_type:
                # ArXiv sometimes returns HTML — follow the redirect
                if "html" in content_type:
                    # Try the direct PDF link
                    pdf_url = url.replace("/abs/", "/pdf/")
                    if pdf_url != url:
                        return download_pdf(pdf_url, dest_path, retries=1)

            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            size_kb = dest_path.stat().st_size / 1024
            if size_kb < 10:
                logger.warning(f"  ⚠ File too small ({size_kb:.1f} KB) — may be invalid")
                dest_path.unlink(missing_ok=True)
                return False

            logger.info(f"  ✓ {dest_path.name} ({size_kb:.0f} KB)")
            return True

        except requests.exceptions.RequestException as e:
            logger.warning(f"  Attempt {attempt} failed: {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)

    logger.error(f"  ✗ Failed to download {dest_path.name}")
    return False


def main():
    parser = argparse.ArgumentParser(description="Download GenAI papers for RAG ingestion")
    parser.add_argument("--max", type=int, default=None, help="Max papers to download")
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    args = parser.parse_args()

    pdf_dir = settings.pdf_data_dir
    logger.info(f"Downloading papers to: {pdf_dir}")
    logger.info(f"Total papers in catalogue: {len(PAPERS)}")

    papers = PAPERS[: args.max] if args.max else PAPERS
    success_count = 0
    skip_count = 0
    fail_count = 0

    for filename, url, description in papers:
        dest_path = pdf_dir / filename

        if dest_path.exists() and not args.force:
            logger.info(f"  ↷ Skipping (exists): {filename}")
            skip_count += 1
            continue

        logger.info(f"\n📄 {description}")
        ok = download_pdf(url, dest_path)
        if ok:
            success_count += 1
        else:
            fail_count += 1

        time.sleep(1.0)  # be polite to ArXiv

    print(f"\n{'='*50}")
    print(f"✓ Downloaded:  {success_count}")
    print(f"↷ Skipped:     {skip_count}")
    print(f"✗ Failed:      {fail_count}")
    print(f"Total in dir:  {len(list(pdf_dir.glob('*.pdf')))}")
    print(f"{'='*50}")
    print("\nNext step: python scripts/ingest_data.py")


if __name__ == "__main__":
    main()
