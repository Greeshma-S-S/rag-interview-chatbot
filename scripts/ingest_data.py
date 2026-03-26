#!/usr/bin/env python3
"""
PDF Ingestion Script — processes PDFs and builds the FAISS vector store.

Usage:
    python scripts/ingest_data.py
    python scripts/ingest_data.py --rebuild   # force rebuild even if store exists
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from app.document_processor import DocumentProcessor
from app.embeddings import EmbeddingManager
from app.vector_store import FAISSVectorStore

try:
    from app.logger import logger
except ImportError:
    from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into FAISS vector store")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of vector store")
    parser.add_argument("--pdf-dir", type=str, default=None, help="Path to PDF directory")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir) if args.pdf_dir else settings.pdf_data_dir
    logger.info(f"PDF directory: {pdf_dir}")
    logger.info(f"Vector store: {settings.vector_store_dir}")

    # Check PDFs exist
    pdfs = list(pdf_dir.glob("*.pdf"))
    if not pdfs:
        logger.error(
            f"No PDFs found in {pdf_dir}. "
            "Run `python scripts/download_pdfs.py` first."
        )
        sys.exit(1)

    logger.info(f"Found {len(pdfs)} PDF(s) to process:")
    for p in pdfs:
        size_kb = p.stat().st_size / 1024
        logger.info(f"  • {p.name} ({size_kb:.0f} KB)")

    # Process documents
    processor = DocumentProcessor()
    documents = processor.load_directory(pdf_dir)

    if not documents:
        logger.error("No documents extracted. Check PDF files.")
        sys.exit(1)

    logger.info(f"\nProcessed {len(documents)} total chunks")

    # Build vector store
    embed_manager = EmbeddingManager(
        cache_dir=settings.vector_store_dir / "embed_cache"
    )
    vector_store = FAISSVectorStore(embed_manager=embed_manager)
    vector_store.build(documents, force_rebuild=args.rebuild)

    logger.info("\n✅ Ingestion complete!")
    logger.info(f"   Documents: {len(documents)} chunks")
    logger.info(f"   Stored at: {settings.vector_store_dir}")
    logger.info("\nNext step: python app/main.py")


if __name__ == "__main__":
    main()
