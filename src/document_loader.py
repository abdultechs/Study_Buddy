import hashlib
import io
import logging
from typing import List, Tuple
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.models import DocumentChunk, DocumentInfo

logger = logging.getLogger(__name__)


class DocumentLoaderError(Exception):
    """Custom exception for PDF validation and loading errors."""
    pass


class DocumentProcessor:
    """Handles PDF validation, hashing, page-by-page text extraction, and chunking."""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    @staticmethod
    def compute_sha256(file_bytes: bytes) -> str:
        """Computes the SHA-256 hash of the uploaded file bytes."""
        return hashlib.sha256(file_bytes).hexdigest()

    def validate_file(self, file_bytes: bytes, filename: str) -> None:
        """Validates file size and PDF magic bytes."""
        if len(file_bytes) == 0:
            raise DocumentLoaderError(f"File '{filename}' is empty (0 bytes).")

        if len(file_bytes) > settings.max_file_size_bytes:
            raise DocumentLoaderError(
                f"File '{filename}' exceeds maximum allowed size of {settings.max_file_size_mb} MB "
                f"({len(file_bytes) / (1024 * 1024):.2f} MB)."
            )

        # Validate PDF magic bytes (%PDF-)
        if not file_bytes.startswith(b"%PDF-"):
            raise DocumentLoaderError(
                f"File '{filename}' does not appear to be a valid PDF (invalid '%PDF-' header)."
            )

    def process_pdf(self, file_bytes: bytes, filename: str) -> Tuple[List[DocumentChunk], DocumentInfo]:
        """Validates and processes a PDF into character chunks with deterministic IDs.
        
        Returns:
            Tuple of (chunks_list, document_info)
        """
        self.validate_file(file_bytes, filename)
        file_hash = self.compute_sha256(file_bytes)

        try:
            reader = PdfReader(io.BytesIO(file_bytes))
        except PdfReadError as e:
            raise DocumentLoaderError(f"Failed to parse PDF '{filename}': {str(e)}")
        except Exception as e:
            raise DocumentLoaderError(f"Corrupt or unreadable PDF '{filename}': {str(e)}")

        total_pages = len(reader.pages)
        if total_pages == 0:
            raise DocumentLoaderError(f"PDF '{filename}' contains no pages.")

        chunks: List[DocumentChunk] = []
        chunk_counter = 0
        extracted_pages = 0

        for page_idx, page in enumerate(reader.pages):
            page_number = page_idx + 1  # 1-indexed page numbering
            try:
                page_text = page.extract_text() or ""
            except Exception as e:
                logger.warning("Could not extract text from page %d of '%s': %s", page_number, filename, e)
                page_text = ""

            page_text = page_text.strip()
            if not page_text:
                continue

            extracted_pages += 1
            # Split page text into chunks
            split_texts = self.text_splitter.split_text(page_text)

            for text_chunk in split_texts:
                chunk_id = f"{file_hash}_c{chunk_counter}"
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=text_chunk,
                        source=filename,
                        page=page_number,
                        file_hash=file_hash,
                        chunk_index=chunk_counter
                    )
                )
                chunk_counter += 1

        if not chunks:
            raise DocumentLoaderError(
                f"PDF '{filename}' contains no selectable text across {total_pages} pages. "
                "The document may be scanned or image-based, which requires OCR."
            )

        doc_info = DocumentInfo(
            file_hash=file_hash,
            filename=filename,
            page_count=total_pages,
            chunk_count=len(chunks)
        )

        return chunks, doc_info
