"""Configuration management using pydantic-settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    database_url: str = "postgresql://coderag:coderag_dev_password@localhost:54329/coderag"

    # Voyage AI
    voyage_api_key: str = ""

    # Embedding Config
    embedding_model_index: str = "voyage-4-large"  # For indexing documents
    embedding_model_query: str = "voyage-4-lite"   # For query embeddings (shared space)
    embedding_dimensions: int = 1024
    embedding_batch_max_tokens: int = 100000  # Voyage limit is 120k, use 100k for safety

    # Two-Pass Retrieval Config
    # Phase 1: Vector Search (cast a wide net)
    retrieval_k: int = 50  # Candidates for reranking (50-100 recommended)
    similarity_threshold: float = 0.30  # Very permissive - let reranker decide quality

    # Phase 2: Rerank
    rerank_model: str = "rerank-2.5"
    rerank_top_k_output: int = 8  # Max final results

    # Phase 3: Relative threshold + Budget
    rerank_relative_factor: float = 0.75  # threshold = top_score * factor
    rerank_score_floor: float = 0.40  # Absolute minimum quality floor
    result_max_tokens: int = 8000  # Token budget for final context

    # Quality logging (JSONL)
    search_log_path: str | None = None  # Path to JSONL file. None = disabled

    # Chunking Config
    chunk_min_tokens: int = 30   # Keep small but meaningful functions
    chunk_max_tokens: int = 600  # Smaller chunks for more precise retrieval
    max_file_size_kb: int = 500  # Skip files larger than this (500KB)
    index_concurrency: int = 10  # Max concurrent files during indexing
    embedding_batch_concurrency: int = 3  # Max concurrent API batches per file
    small_file_lines: int = 200  # Files with fewer lines are kept as single chunk (not fragmented)

    # File type filter - languages considered "code" (vs docs/config)
    code_languages: tuple[str, ...] = (
        "typescript", "javascript", "python", "java", "go", "rust", "sql"
    )
    docs_languages: tuple[str, ...] = ("markdown",)

    # Server Config
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
