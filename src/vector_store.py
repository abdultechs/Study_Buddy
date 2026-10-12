import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings
from src.models import DocumentChunk, DocumentInfo

logger = logging.getLogger(__name__)

COLLECTION_NAME = "pdf_knowledge_base"


class EmbeddingModelMismatchError(Exception):
    """Raised when the loaded collection was created with a different embedding model."""
    pass


class VectorStoreManager:
    """Manages persistent ChromaDB vector storage, embeddings, upserts, and search."""

    def __init__(self, persist_directory: Optional[str] = None):
        self.persist_directory = persist_directory or settings.chroma_persist_dir
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        self.embeddings = self._init_embeddings()
        self.chroma_client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self._init_collection()
        self.vector_store = Chroma(
            client=self.chroma_client,
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
        )

    def _init_embeddings(self):
        """Initializes embeddings model based on environment configuration."""
        if settings.embedding_provider == "gemini":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            if not settings.gemini_api_key:
                raise ValueError("GEMINI_API_KEY is required when EMBEDDING_PROVIDER=gemini")
            return GoogleGenerativeAIEmbeddings(
                model=settings.embedding_model,
                google_api_key=settings.gemini_api_key
            )
        else:
            # Default: Local HuggingFace sentence-transformers
            return HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )

    def _init_collection(self):
        """Creates or retrieves Chroma collection, verifying embedding model metadata."""
        # Check existing collection
        existing_collections = [c.name for c in self.chroma_client.list_collections()]
        expected_metadata = {
            "hnsw:space": "cosine",
            "embedding_model": settings.embedding_model
        }

        if COLLECTION_NAME in existing_collections:
            collection = self.chroma_client.get_collection(COLLECTION_NAME)
            stored_model = (collection.metadata or {}).get("embedding_model")
            if stored_model and stored_model != settings.embedding_model:
                raise EmbeddingModelMismatchError(
                    f"Collection was indexed using '{stored_model}', but current config is "
                    f"'{settings.embedding_model}'. Reset the vector store or revert EMBEDDING_MODEL in .env."
                )
            return collection
        else:
            return self.chroma_client.create_collection(
                name=COLLECTION_NAME,
                metadata=expected_metadata
            )

    def is_file_indexed(self, file_hash: str) -> bool:
        """Checks if any chunks with this file_hash already exist in the vector store."""
        results = self.collection.get(
            where={"file_hash": file_hash},
            limit=1
        )
        return bool(results and results.get("ids"))

    def upsert_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Deterministically upserts document chunks into ChromaDB.
        
        Returns:
            Number of chunks upserted.
        """
        if not chunks:
            return 0

        # Check if already indexed
        file_hash = chunks[0].file_hash
        if self.is_file_indexed(file_hash):
            logger.info("File with hash %s is already indexed. Upserting to ensure parity.", file_hash[:12])

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "source": c.source,
                "page": int(c.page),
                "file_hash": c.file_hash,
                "chunk_index": int(c.chunk_index)
            }
            for c in chunks
        ]

        # Embed and upsert
        embeddings = self.embeddings.embed_documents(documents)
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        return len(chunks)

    def search_with_similarity(
        self,
        query: str,
        top_k: int = 4,
        file_hash_filter: Optional[str] = None
    ) -> List[Tuple[Document, float]]:
        """Queries Chroma collection and returns Documents with cosine similarity scores.
        
        Formula: similarity = 1 - distance (cosine distance where d in [0, 2])
        """
        query_embedding = self.embeddings.embed_query(query)
        where_filter = {"file_hash": file_hash_filter} if file_hash_filter else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        docs_with_scores: List[Tuple[Document, float]] = []
        if not results or not results["documents"] or not results["documents"][0]:
            return docs_with_scores

        docs = results["documents"][0]
        metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
        distances = results["distances"][0] if results.get("distances") else [1.0] * len(docs)

        for doc_text, meta, distance in zip(docs, metas, distances):
            # Exact formula: similarity = 1 - distance
            similarity = 1.0 - float(distance)
            doc = Document(page_content=doc_text, metadata=meta)
            docs_with_scores.append((doc, similarity))

        return docs_with_scores

    def get_indexed_documents(self) -> List[DocumentInfo]:
        """Derives the active document list dynamically from Chroma metadata."""
        all_records = self.collection.get(include=["metadatas"])
        if not all_records or not all_records.get("metadatas"):
            return []

        doc_summary: Dict[str, Dict] = {}
        for meta in all_records["metadatas"]:
            f_hash = meta.get("file_hash", "unknown")
            source = meta.get("source", "unknown")
            page = meta.get("page", 1)

            if f_hash not in doc_summary:
                doc_summary[f_hash] = {
                    "filename": source,
                    "pages": set(),
                    "chunk_count": 0
                }
            doc_summary[f_hash]["pages"].add(page)
            doc_summary[f_hash]["chunk_count"] += 1

        return [
            DocumentInfo(
                file_hash=f_hash,
                filename=info["filename"],
                page_count=len(info["pages"]),
                chunk_count=info["chunk_count"]
            )
            for f_hash, info in doc_summary.items()
        ]

    def reset(self) -> None:
        """Deletes all collections and clears the persistent database directory."""
        try:
            self.chroma_client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        # Re-initialize empty collection
        expected_metadata = {
            "hnsw:space": "cosine",
            "embedding_model": settings.embedding_model
        }
        self.collection = self.chroma_client.create_collection(
            name=COLLECTION_NAME,
            metadata=expected_metadata
        )
        self.vector_store = Chroma(
            client=self.chroma_client,
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
        )
