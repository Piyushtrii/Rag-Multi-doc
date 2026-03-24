from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from loguru import logger

from app.core.config import settings

# Optional heavy imports (guard so the rest of app still loads if missing)
try:
    from pypdf import PdfReader

    _PYPDF_OK = True
except ImportError:  # pragma: no cover
    _PYPDF_OK = False
    logger.warning("pypdf not installed — PDF files will be skipped")

try:
    import docx as _docx  # python-docx

    _DOCX_OK = True
except ImportError:  # pragma: no cover
    _DOCX_OK = False
    logger.warning("python-docx not installed — DOCX files will be skipped")


_SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf", ".docx"}


class DocumentLoader:
    """Load and chunk documents from the ``docs/`` folder.

    Supported formats: TXT, Markdown, PDF, DOCX.
    """

    def __init__(self, docs_path: Path | None = None) -> None:
        self.docs_path = docs_path or settings.DOCS_PATH
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP

    #Public API 

    def load_documents(self) -> List[Dict]:
        """Return a list of raw document dicts from all supported files."""
        documents: List[Dict] = []

        if not self.docs_path.exists():
            logger.warning(f"Docs path does not exist: {self.docs_path}")
            return documents

        # rglob("*") walks sub-directories as well
        for file_path in sorted(self.docs_path.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in _SUPPORTED_SUFFIXES:
                logger.debug(f"Skipping unsupported file: {file_path.name}")
                continue

            try:
                content = self._read_file(file_path)
                if not content.strip():
                    logger.warning(f"Empty content in {file_path.name}, skipping")
                    continue

                documents.append(
                    {
                        "id": file_path.stem,
                        "source": file_path.name,
                        "content": content,
                        "metadata": {
                            "source": file_path.name,
                            "doc_id": file_path.stem,
                            "file_type": file_path.suffix.lower().lstrip("."),
                            "file_path": str(file_path),
                        },
                    }
                )
                logger.info(f"Loaded [{file_path.suffix.upper()}] {file_path.name}")

            except Exception as exc:
                logger.error(f"Error loading {file_path.name}: {exc}")

        logger.info(f"Loaded {len(documents)} documents total")
        return documents

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """Split each document into overlapping chunks."""
        chunked: List[Dict] = []

        for doc in documents:
            chunks = self._chunk_text(doc["content"])
            for i, chunk in enumerate(chunks):
                chunked.append(
                    {
                        "id": f"{doc['id']}_chunk_{i}",
                        "content": chunk,
                        "source": doc["source"],
                        "metadata": {
                            **doc["metadata"],
                            "chunk_id": i,
                        },
                    }
                )

        logger.info(f"Created {len(chunked)} chunks from {len(documents)} documents")
        return chunked

    #Format reader 

    def _read_file(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()
        if suffix in {".txt", ".md"}:
            return self._read_text(file_path)
        if suffix == ".pdf":
            return self._read_pdf(file_path)
        if suffix == ".docx":
            return self._read_docx(file_path)
        raise ValueError(f"Unsupported file type: {suffix}")

    @staticmethod
    def _read_text(file_path: Path) -> str:
        return file_path.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def _read_pdf(file_path: Path) -> str:
        if not _PYPDF_OK:
            raise RuntimeError("pypdf is not installed; cannot read PDF files")
        try:
            reader = PdfReader(str(file_path))
            pages = []
            for i, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                    if text:
                        pages.append(text)
                except Exception as page_e:
                    logger.warning(f"Could not extract text from page {i} of {file_path.name}: {page_e}")
            return "\n\n".join(pages)
        except Exception as e:
            logger.error(f"Failed to read PDF {file_path.name} entirely: {e}")
            return ""

    @staticmethod
    def _read_docx(file_path: Path) -> str:
        if not _DOCX_OK:
            raise RuntimeError("python-docx is not installed; cannot read DOCX files")
        doc = _docx.Document(str(file_path))
        return "\n".join(para.text for para in doc.paragraphs if para.text.strip())

    #Chunking

    def _chunk_text(self, content: str) -> List[str]:
        """Split *content* into sentence-aware chunks with overlap."""
        # Normalise newlines and split on sentence boundaries
        sentences = content.replace("\n", " ").split(". ")

        chunks: List[str] = []
        current_chunk = ""
        overlap_buffer = ""

        for sentence in sentences:
            candidate = sentence + ". "
            if len(current_chunk) + len(candidate) <= self.chunk_size:
                current_chunk += candidate
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # Keep last CHUNK_OVERLAP characters as overlap seed
                    overlap = int(self.chunk_overlap)
                    overlap_buffer = current_chunk[-overlap:] if overlap > 0 else ""
                current_chunk = overlap_buffer + candidate
                overlap_buffer = ""

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks
