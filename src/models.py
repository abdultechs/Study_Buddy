from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SourceCitation:
    """Represents a citation referencing a specific page in a document."""
    file_name: str
    page_number: int
    file_hash: str
    snippet: str


@dataclass
class ChatResponse:
    """Represents the complete response from the RAG pipeline."""
    answer: str
    citations: List[SourceCitation] = field(default_factory=list)
    is_fallback: bool = False
    max_similarity: float = 0.0


@dataclass
class DocumentChunk:
    """Represents an individual text chunk with strict metadata tracking."""
    chunk_id: str
    text: str
    source: str
    page: int
    file_hash: str
    chunk_index: int


@dataclass
class DocumentInfo:
    """Summary metadata for an indexed document in the vector store."""
    file_hash: str
    filename: str
    page_count: int
    chunk_count: int
