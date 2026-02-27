"""Book indexer for embedding and storing book chunks."""

import logging
from pathlib import Path

from code_context.books.parser import EpubParser, TextChunk, generate_book_id
from code_context.db.pool import DatabasePool
from code_context.embedding.voyage import VoyageClient

logger = logging.getLogger(__name__)


class BookIndexer:
    """Index books into pgvector for semantic search."""

    def __init__(self, db: DatabasePool, voyage: VoyageClient):
        self.db = db
        self.voyage = voyage
        self.parser = EpubParser(max_chunk_tokens=500, overlap_tokens=50)

    async def index_book(
        self, filepath: Path, *, force: bool = False, category: str = "software"
    ) -> dict:
        """Index a single book.

        Returns:
            Dict with book_id, title, chunks_indexed, skipped.
        """
        # Parse just enough to get the book_id
        metadata = self.parser.parse(filepath)
        book_id = generate_book_id(metadata.title)

        # Skip if already indexed (unless forced)
        if not force and await self._is_indexed(book_id):
            logger.info(f"Skipping (already indexed): {metadata.title}")
            return {
                "book_id": book_id,
                "title": metadata.title,
                "author": metadata.author,
                "skipped": True,
            }

        logger.info(f"Indexing: {metadata.title} by {metadata.author}")
        logger.info(f"Chapters: {len(metadata.chapters)}")

        chunks = self.parser.chunk_book(metadata)
        logger.info(f"Generated {len(chunks)} chunks")

        # Delete existing data for this book
        await self._delete_book(book_id)

        # Insert book metadata
        await self._insert_book(
            book_id=book_id,
            title=metadata.title,
            author=metadata.author,
            filepath=str(filepath),
            chapter_count=len(metadata.chapters),
            category=category,
        )

        # Generate embeddings and insert chunks
        if chunks:
            texts = [c.text for c in chunks]
            logger.info(f"Generating embeddings for {len(texts)} chunks...")

            embeddings = await self.voyage.embed_documents(texts, input_type="document")

            logger.info(f"Inserting {len(chunks)} chunks into database...")
            await self._insert_chunks(book_id, chunks, embeddings)

        return {
            "book_id": book_id,
            "title": metadata.title,
            "author": metadata.author,
            "chapters": len(metadata.chapters),
            "chunks_indexed": len(chunks),
            "skipped": False,
        }

    async def index_directory(
        self, directory: Path, *, force: bool = False, category: str = "software"
    ) -> list[dict]:
        """Index all EPUB files in a directory."""
        results = []
        epub_files = list(directory.glob("*.epub"))

        logger.info(f"Found {len(epub_files)} EPUB files in {directory}")

        for filepath in epub_files:
            try:
                result = await self.index_book(
                    filepath, force=force, category=category
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to index {filepath.name}: {e}")
                results.append({"filepath": str(filepath), "error": str(e)})

        return results

    async def create_vector_index(self) -> None:
        """Create the vector index for book chunks."""
        async with self.db.acquire() as conn:
            await conn.execute("SELECT create_book_vector_index()")
            logger.info("Book vector index created")

    async def _is_indexed(self, book_id: str) -> bool:
        """Check if a book is already indexed."""
        async with self.db.acquire() as conn:
            row = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM books WHERE id = $1)", book_id
            )
            return row

    async def _delete_book(self, book_id: str) -> None:
        """Delete a book and its chunks."""
        async with self.db.acquire() as conn:
            await conn.execute("DELETE FROM books WHERE id = $1", book_id)

    async def _insert_book(
        self,
        book_id: str,
        title: str,
        author: str,
        filepath: str,
        chapter_count: int,
        category: str = "software",
    ) -> None:
        """Insert book metadata."""
        async with self.db.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO books (id, title, author, filepath, category, chapter_count)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (id) DO UPDATE SET
                    title = EXCLUDED.title,
                    author = EXCLUDED.author,
                    filepath = EXCLUDED.filepath,
                    category = EXCLUDED.category,
                    chapter_count = EXCLUDED.chapter_count,
                    indexed_at = NOW()
                """,
                book_id,
                title,
                author,
                filepath,
                category,
                chapter_count,
            )

    async def _insert_chunks(
        self,
        book_id: str,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Insert book chunks with embeddings."""
        async with self.db.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO book_chunks
                    (book_id, chapter_title, section, chunk_text, embedding, chunk_index, token_count)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                [
                    (
                        book_id,
                        chunk.chapter_title,
                        chunk.section,
                        chunk.text,
                        embedding,
                        chunk.chunk_index,
                        chunk.token_count,
                    )
                    for chunk, embedding in zip(chunks, embeddings)
                ],
            )
