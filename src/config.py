import os
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Automatically load .env file from project root if present
load_dotenv()

# Ensure HuggingFace Hub uses reliable mirror on networks with CDN restrictions
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


class Settings(BaseSettings):
    """Central configuration management using Pydantic Settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # LLM Settings (Verified 2026 models: gemini-3.8-flash, gemini-3.5-flash)
    llm_provider: str = Field(default="gemini", validation_alias="LLM_PROVIDER")
    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-3.8-flash", validation_alias="GEMINI_MODEL")
    gemini_temperature: float = Field(default=0.0, validation_alias="GEMINI_TEMPERATURE")

    # Optional OpenAI Settings
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", validation_alias="OPENAI_MODEL")

    # Embedding Settings (Local sentence-transformers or Cloud gemini-embedding-001)
    embedding_provider: Literal["local", "gemini"] = Field(
        default="local", validation_alias="EMBEDDING_PROVIDER"
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2", validation_alias="EMBEDDING_MODEL"
    )

    # RAG & Chunking Parameters
    chunk_size: int = Field(default=800, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, validation_alias="CHUNK_OVERLAP")
    top_k_retrieval: int = Field(default=4, validation_alias="TOP_K_RETRIEVAL")
    relevance_threshold: float = Field(default=0.60, validation_alias="RELEVANCE_THRESHOLD")
    history_window_messages: int = Field(default=6, validation_alias="HISTORY_WINDOW_MESSAGES")

    # File Upload Controls
    max_file_size_mb: int = Field(default=20, validation_alias="MAX_FILE_SIZE_MB")

    # Storage Paths
    chroma_persist_dir: str = Field(default="./data/chroma_db", validation_alias="CHROMA_PERSIST_DIR")

    @property
    def max_file_size_bytes(self) -> int:
        """Returns maximum allowed file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024

    def ensure_directories(self) -> None:
        """Ensures that required data directories exist."""
        Path(self.chroma_persist_dir).mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
