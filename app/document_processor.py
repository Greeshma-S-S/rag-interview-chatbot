"""
Document processing module.
Loads PDFs, splits them into chunks, and returns LangChain Documents.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader  # type: ignore

from config import settings

try:
    from app.logger import logger
except ImportError:
    from logger import logger


class DocumentProcessor:
    """
    Processes PDF files into chunked LangChain Documents ready for embedding.
    """

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            length_function=len,
        )
        logger.info(
            f"DocumentProcessor — chunk_size={settings.chunk_size}, "
            f"overlap={settings.chunk_overlap}"
        )

    # ── Public API ────────────────────────────────────────────

    def load_pdf(self, pdf_path: Path) -> List[Document]:
        """Load a single PDF and return a list of chunked Documents."""
        logger.info(f"Loading PDF: {pdf_path.name}")
        raw_text = self._extract_text(pdf_path)
        if not raw_text.strip():
            logger.warning(f"No text extracted from {pdf_path.name}")
            return []

        chunks = self.splitter.split_text(raw_text)
        documents = [
            Document(
                page_content=chunk,
                metadata={
                    "source": pdf_path.name,
                    "source_path": str(pdf_path),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                },
            )
            for i, chunk in enumerate(chunks)
        ]
        logger.info(f"  → {len(documents)} chunks from {pdf_path.name}")
        return documents

    def load_directory(self, directory: Path) -> List[Document]:
        """Load all PDFs in a directory and return all chunked Documents."""
        pdfs = list(directory.glob("*.pdf"))
        if not pdfs:
            logger.warning(f"No PDFs found in {directory}")
            return []

        all_docs: List[Document] = []
        for pdf_path in pdfs:
            docs = self.load_pdf(pdf_path)
            all_docs.extend(docs)

        logger.info(f"Total: {len(all_docs)} chunks from {len(pdfs)} PDFs")
        return all_docs

    # ── Internal helpers ──────────────────────────────────────

    def _extract_text(self, pdf_path: Path) -> str:
        """Extract raw text from a PDF file."""
        try:
            reader = PdfReader(str(pdf_path))
            pages_text = []
            for page in reader.pages:
                text = page.extract_text() or ""
                pages_text.append(text)
            raw = "\n\n".join(pages_text)
            return self._clean_text(raw)
        except Exception as exc:
            logger.error(f"Failed to extract text from {pdf_path.name}: {exc}")
            return ""

    @staticmethod
    def _clean_text(text: str) -> str:
        """Basic text normalisation."""
        # Remove excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        # Remove page numbers / headers that are just numbers
        text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)
        return text.strip()
