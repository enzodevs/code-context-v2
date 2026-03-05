"""Async database connection pool management."""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import asyncpg
from pgvector.asyncpg import register_vector

from code_context.config import get_settings
from code_context.db.schema import ChunkResult, CodeChunk, CodeFile

logger = logging.getLogger(__name__)


class DatabasePool:
    """Manages asyncpg connection pool with pgvector support."""

    def __init__(self, dsn: str | None = None, min_size: int = 5, max_size: int = 20):
        self.dsn = dsn or get_settings().database_url
        self._min_size = min_size
        self._max_size = max_size
        self._pool: asyncpg.Pool | None = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the connection pool."""
        if self._pool is not None:
            return

        async with self._lock:
            if self._pool is not None:
                return

            async def init_connection(conn: asyncpg.Connection) -> None:
                await register_vector(conn)

            self._pool = await asyncpg.create_pool(
                self.dsn,
                min_size=self._min_size,
                max_size=self._max_size,
                max_inactive_connection_lifetime=300,
                init=init_connection,
            )
            logger.info("Database pool initialized")

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Database pool closed")

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[asyncpg.Connection]:
        """Acquire a connection from the pool."""
        if self._pool is None:
            await self.initialize()
        async with self._pool.acquire() as conn:  # type: ignore
            yield conn

    # File operations
    async def upsert_file(self, file: CodeFile) -> None:
        """Insert or update a file record."""
        async with self.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO code_files (filepath, file_hash, language, project_root, project_id, last_indexed, loc)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (filepath) DO UPDATE SET
                    file_hash = EXCLUDED.file_hash,
                    last_indexed = EXCLUDED.last_indexed,
                    loc = EXCLUDED.loc
                """,
                file.filepath,
                file.file_hash,
                file.language,
                file.project_root,
                file.project_id,
                file.last_indexed,
                file.loc,
            )

    async def get_file(self, filepath: str) -> CodeFile | None:
        """Get file by filepath."""
        async with self.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM code_files WHERE filepath = $1", filepath
            )
            if row:
                return CodeFile(**dict(row))
            return None

    async def get_project_root(self, project_id: str) -> str | None:
        """Get the root path for a project by its ID."""
        async with self.acquire() as conn:
            return await conn.fetchval(
                "SELECT project_root FROM code_files WHERE project_id = $1 LIMIT 1",
                project_id,
            )

    async def get_project_files(self, project_id: str) -> dict[str, str]:
        """Get all indexed files for a project as {filepath: file_hash}."""
        async with self.acquire() as conn:
            rows = await conn.fetch(
                "SELECT filepath, file_hash FROM code_files WHERE project_id = $1",
                project_id,
            )
            return {row["filepath"]: row["file_hash"] for row in rows}

    async def delete_file(self, filepath: str) -> None:
        """Delete file and its chunks (cascade)."""
        async with self.acquire() as conn:
            await conn.execute("DELETE FROM code_files WHERE filepath = $1", filepath)

    async def delete_files(self, filepaths: list[str]) -> int:
        """Delete multiple files (and cascading chunks) in a single query."""
        if not filepaths:
            return 0

        async with self.acquire() as conn:
            deleted = await conn.fetchval(
                """
                WITH deleted AS (
                    DELETE FROM code_files
                    WHERE filepath = ANY($1::text[])
                    RETURNING 1
                )
                SELECT COUNT(*) FROM deleted
                """,
                filepaths,
            )
            return int(deleted or 0)

    # Chunk operations
    async def insert_chunks(self, chunks: list[CodeChunk]) -> None:
        """Bulk insert chunks."""
        if not chunks:
            return

        async with self.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO code_chunks
                    (filepath, chunk_hash, chunk_text, embedding, start_line, end_line,
                     chunk_type, symbol_name, context_metadata, token_count)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10)
                """,
                [
                    (
                        c.filepath,
                        c.chunk_hash,
                        c.chunk_text,
                        c.embedding,
                        c.start_line,
                        c.end_line,
                        c.chunk_type,
                        c.symbol_name,
                        json.dumps(c.context_metadata),  # Convert dict to JSON string
                        c.token_count,
                    )
                    for c in chunks
                ],
            )

    async def reindex_file_atomic(self, file: CodeFile, chunks: list[CodeChunk]) -> None:
        """Atomically reindex a file: delete old chunks, upsert file, insert new chunks.

        All operations run in a single transaction. If interrupted (e.g. Ctrl+C),
        the database rolls back to the previous consistent state.
        """
        async with self.acquire() as conn:
            async with conn.transaction():
                # 1. Delete old chunks
                await conn.execute(
                    "DELETE FROM code_chunks WHERE filepath = $1", file.filepath
                )

                # 2. Upsert file record
                await conn.execute(
                    """
                    INSERT INTO code_files (filepath, file_hash, language, project_root, project_id, last_indexed, loc)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (filepath) DO UPDATE SET
                        file_hash = EXCLUDED.file_hash,
                        last_indexed = EXCLUDED.last_indexed,
                        loc = EXCLUDED.loc
                    """,
                    file.filepath,
                    file.file_hash,
                    file.language,
                    file.project_root,
                    file.project_id,
                    file.last_indexed,
                    file.loc,
                )

                # 3. Insert new chunks
                if chunks:
                    await conn.executemany(
                        """
                        INSERT INTO code_chunks
                            (filepath, chunk_hash, chunk_text, embedding, start_line, end_line,
                             chunk_type, symbol_name, context_metadata, token_count)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10)
                        """,
                        [
                            (
                                c.filepath,
                                c.chunk_hash,
                                c.chunk_text,
                                c.embedding,
                                c.start_line,
                                c.end_line,
                                c.chunk_type,
                                c.symbol_name,
                                json.dumps(c.context_metadata),
                                c.token_count,
                            )
                            for c in chunks
                        ],
                    )

    async def delete_chunks_for_file(self, filepath: str) -> int:
        """Delete all chunks for a file."""
        async with self.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM code_chunks WHERE filepath = $1", filepath
            )
            return int(result.split()[-1])

    async def search_chunks(
        self,
        query_embedding: list[float],
        filepath: str | None = None,
        chunk_type: str | None = None,
        language: str | None = None,
        limit: int = 20,
        project: str | None = None,
        languages: list[str] | None = None,
        directory: str | None = None,
    ) -> list[ChunkResult]:
        """Vector similarity search with optional filtering.

        Args:
            project: Can be either the short project ID (folder name) or the full path.
            languages: Filter to only these languages (for file_type='code' filtering).
            directory: Filter to files within this directory path prefix.
        """
        async with self.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM search_chunks($1::vector, $2, $3, $4, $5, $6, $7, $8)
                """,
                query_embedding,
                filepath,
                chunk_type,
                language,
                limit,
                project,
                languages,
                directory,
            )
            return [
                ChunkResult(
                    chunk_id=row["chunk_id"],
                    filepath=row["filepath"],
                    chunk_text=row["chunk_text"],
                    chunk_type=row["chunk_type"],
                    symbol_name=row["symbol_name"],
                    start_line=row["start_line"],
                    end_line=row["end_line"],
                    context_metadata=(
                        json.loads(row["context_metadata"])
                        if isinstance(row["context_metadata"], str)
                        else row["context_metadata"] or {}
                    ),
                    similarity=row["similarity"],
                )
                for row in rows
            ]

    async def get_index_stats(self) -> dict:
        """Get indexing statistics."""
        async with self.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM index_stats")
            stats = {
                "total_files": 0,
                "total_chunks": 0,
                "total_projects": 0,
                "by_language": {},
            }
            for row in rows:
                lang = row["language"]
                stats["total_files"] += row["total_files"] or 0
                stats["total_chunks"] += row["total_chunks"] or 0
                stats["total_projects"] = max(
                    stats["total_projects"], row["total_projects"] or 0
                )
                stats["by_language"][lang] = {
                    "files": row["total_files"],
                    "chunks": row["total_chunks"],
                    "files_chunks": row.get("files", 0),
                    "declarations": row.get("declarations", 0),
                    "functions": row["functions"],
                    "classes": row["classes"],
                    "methods": row["methods"],
                    "avg_tokens": row["avg_tokens_per_chunk"],
                }
            return stats

    async def get_quality_stats(self, project_id: str | None = None) -> dict:
        """Get chunk quality indicators (signature coverage, tiny/oversized counts)."""
        async with self.acquire() as conn:
            where = "WHERE f.project_id = $1" if project_id else ""
            args: list = [project_id] if project_id else []

            row = await conn.fetchrow(
                f"""
                SELECT
                    COUNT(*) FILTER (WHERE c.chunk_type IN ('function', 'method'))
                        AS callable_chunks,
                    COUNT(*) FILTER (
                        WHERE c.chunk_type IN ('function', 'method')
                        AND c.symbol_name IS NOT NULL
                        AND c.symbol_name != ''
                    ) AS with_signature,
                    COUNT(*) FILTER (WHERE c.token_count < 30) AS tiny_chunks,
                    COUNT(*) FILTER (WHERE c.token_count > 600) AS oversized_chunks,
                    COUNT(*) FILTER (WHERE c.chunk_type = 'file') AS file_chunks,
                    COUNT(*) AS total_chunks
                FROM code_chunks c
                JOIN code_files f ON c.filepath = f.filepath
                {where}
                """,
                *args,
            )
            return dict(row) if row else {}

    async def create_vector_index(self) -> None:
        """Ensure the vector index exists.

        Uses IF NOT EXISTS to avoid expensive rebuilds during normal sync.
        """
        async with self.acquire() as conn:
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON code_chunks
                USING diskann (embedding)
                WITH (num_neighbors = 50, search_list_size = 100, max_alpha = 1.2)
                """
            )
            logger.info("Vector index ensured")

    # Project management operations
    async def list_projects(self) -> list[dict]:
        """List all indexed projects with stats."""
        async with self.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    project_id,
                    project_root,
                    COUNT(DISTINCT f.filepath) as file_count,
                    COUNT(c.id) as chunk_count,
                    MAX(f.last_indexed) as last_indexed,
                    SUM(f.loc) as total_loc,
                    array_agg(DISTINCT f.language) as languages
                FROM code_files f
                LEFT JOIN code_chunks c ON f.filepath = c.filepath
                GROUP BY project_id, project_root
                ORDER BY last_indexed DESC
                """
            )
            return [
                {
                    "project_id": row["project_id"],
                    "project_root": row["project_root"],
                    "file_count": row["file_count"],
                    "chunk_count": row["chunk_count"],
                    "last_indexed": row["last_indexed"],
                    "total_loc": row["total_loc"] or 0,
                    "languages": [l for l in row["languages"] if l],
                }
                for row in rows
            ]

    async def get_project_stats(self, project_id: str) -> dict | None:
        """Get detailed stats for a specific project."""
        async with self.acquire() as conn:
            # Basic stats
            row = await conn.fetchrow(
                """
                SELECT
                    project_id,
                    project_root,
                    COUNT(DISTINCT f.filepath) as file_count,
                    COUNT(c.id) as chunk_count,
                    MAX(f.last_indexed) as last_indexed,
                    SUM(f.loc) as total_loc
                FROM code_files f
                LEFT JOIN code_chunks c ON f.filepath = c.filepath
                WHERE f.project_id = $1
                GROUP BY f.project_id, f.project_root
                """,
                project_id,
            )
            if not row or row["file_count"] == 0:
                return None

            # Language breakdown with hierarchical chunk types
            lang_rows = await conn.fetch(
                """
                SELECT
                    f.language,
                    COUNT(DISTINCT f.filepath) as files,
                    COUNT(c.id) as chunks,
                    COUNT(CASE WHEN c.chunk_type = 'file' THEN 1 END) as files_chunks,
                    COUNT(CASE WHEN c.chunk_type = 'declaration' THEN 1 END) as declarations,
                    COUNT(CASE WHEN c.chunk_type = 'function' THEN 1 END) as functions,
                    COUNT(CASE WHEN c.chunk_type = 'class' THEN 1 END) as classes,
                    COUNT(CASE WHEN c.chunk_type = 'method' THEN 1 END) as methods
                FROM code_files f
                LEFT JOIN code_chunks c ON f.filepath = c.filepath
                WHERE f.project_id = $1
                GROUP BY f.language
                """,
                project_id,
            )

            return {
                "project_id": row["project_id"],
                "project_root": row["project_root"],
                "file_count": row["file_count"],
                "chunk_count": row["chunk_count"],
                "last_indexed": row["last_indexed"],
                "total_loc": row["total_loc"] or 0,
                "by_language": {
                    r["language"]: {
                        "files": r["files"],
                        "chunks": r["chunks"],
                        "files_chunks": r["files_chunks"],
                        "declarations": r["declarations"],
                        "functions": r["functions"],
                        "classes": r["classes"],
                        "methods": r["methods"],
                    }
                    for r in lang_rows
                },
            }

    async def delete_project(self, project_id: str) -> dict:
        """Delete all files and chunks for a project."""
        async with self.acquire() as conn:
            # Get counts before deletion
            counts = await conn.fetchrow(
                """
                SELECT
                    COUNT(DISTINCT f.filepath) as files,
                    COUNT(c.id) as chunks
                FROM code_files f
                LEFT JOIN code_chunks c ON f.filepath = c.filepath
                WHERE f.project_id = $1
                """,
                project_id,
            )

            # Delete (chunks cascade)
            await conn.execute(
                "DELETE FROM code_files WHERE project_id = $1", project_id
            )

            return {
                "deleted_files": counts["files"] or 0,
                "deleted_chunks": counts["chunks"] or 0,
            }


    # Book operations
    async def list_books(self) -> list[dict]:
        """List all indexed books."""
        async with self.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    b.id,
                    b.title,
                    b.author,
                    b.category,
                    b.chapter_count,
                    COUNT(bc.id) as chunk_count,
                    b.indexed_at
                FROM books b
                LEFT JOIN book_chunks bc ON b.id = bc.book_id
                GROUP BY b.id
                ORDER BY b.indexed_at DESC
                """
            )
            return [
                {
                    "id": row["id"],
                    "title": row["title"],
                    "author": row["author"],
                    "category": row["category"],
                    "chapter_count": row["chapter_count"],
                    "chunk_count": row["chunk_count"],
                }
                for row in rows
            ]

    async def search_book_chunks(
        self,
        query_embedding: list[float],
        book_id: str | None = None,
        limit: int = 5,
        category: str | None = None,
    ) -> list[dict]:
        """Search book chunks by vector similarity."""
        async with self.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM search_book_chunks($1::vector, $2, $3, $4)",
                query_embedding,
                book_id,
                limit,
                category,
            )
            return [
                {
                    "chunk_id": row["chunk_id"],
                    "book_id": row["book_id"],
                    "book_title": row["book_title"],
                    "author": row["author"],
                    "chapter": row["chapter_title"],
                    "section": row["section"],
                    "content": row["chunk_text"],
                    "relevance": row["similarity"],
                }
                for row in rows
            ]


# Global pool instance
_pool: DatabasePool | None = None


async def get_db_pool() -> DatabasePool:
    """Get or create the global database pool."""
    global _pool
    if _pool is None:
        _pool = DatabasePool()
        await _pool.initialize()
    return _pool
