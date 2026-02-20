"""Pydantic models for database entities."""

from datetime import datetime

from pydantic import BaseModel, Field


class CodeFile(BaseModel):
    """Represents a tracked code file."""

    filepath: str
    file_hash: str
    language: str
    project_root: str
    project_id: str  # Short identifier for the project
    last_indexed: datetime = Field(default_factory=datetime.now)
    loc: int = 0


class CodeChunk(BaseModel):
    """Represents a code chunk with embedding."""

    id: int | None = None
    filepath: str
    chunk_hash: str
    chunk_text: str
    embedding: list[float] | None = None
    start_line: int
    end_line: int
    chunk_type: str  # function | class | method | module | import
    symbol_name: str | None = None
    context_metadata: dict = Field(default_factory=dict)
    token_count: int = 0
    created_at: datetime = Field(default_factory=datetime.now)


class ChunkResult(BaseModel):
    """Search result with similarity score."""

    chunk_id: int
    filepath: str
    chunk_text: str
    chunk_type: str
    symbol_name: str | None
    start_line: int
    end_line: int
    context_metadata: dict
    similarity: float
    relevance_score: float | None = None  # After reranking
