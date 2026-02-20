-- Books Literature Search - Database Schema
-- Extends the existing pgvector setup

-- Books metadata table
CREATE TABLE IF NOT EXISTS books (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    author TEXT NOT NULL,
    filepath TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'software',
    chapter_count INTEGER DEFAULT 0,
    indexed_at TIMESTAMPTZ DEFAULT NOW()
);

-- Book chunks with embeddings
CREATE TABLE IF NOT EXISTS book_chunks (
    id BIGSERIAL PRIMARY KEY,
    book_id TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    chapter_title TEXT NOT NULL,
    section TEXT,
    chunk_text TEXT NOT NULL,
    embedding vector(1024),
    chunk_index INTEGER NOT NULL,
    token_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_book_chunks_book_id ON book_chunks(book_id);
CREATE INDEX IF NOT EXISTS idx_book_chunks_chapter ON book_chunks(chapter_title);

-- Vector search function for books
CREATE OR REPLACE FUNCTION search_book_chunks(
    query_embedding vector(1024),
    p_book_id TEXT DEFAULT NULL,
    p_limit INTEGER DEFAULT 5,
    p_category TEXT DEFAULT NULL
)
RETURNS TABLE (
    chunk_id BIGINT,
    book_id TEXT,
    book_title TEXT,
    author TEXT,
    chapter_title TEXT,
    section TEXT,
    chunk_text TEXT,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        bc.id,
        bc.book_id,
        b.title,
        b.author,
        bc.chapter_title,
        bc.section,
        bc.chunk_text,
        1 - (bc.embedding <=> query_embedding) AS similarity
    FROM book_chunks bc
    JOIN books b ON bc.book_id = b.id
    WHERE
        (p_book_id IS NULL OR bc.book_id = p_book_id)
        AND (p_category IS NULL OR b.category = p_category)
        AND bc.embedding IS NOT NULL
    ORDER BY bc.embedding <=> query_embedding
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Create diskann index (call after data insertion)
CREATE OR REPLACE FUNCTION create_book_vector_index()
RETURNS void AS $$
BEGIN
    DROP INDEX IF EXISTS idx_book_chunks_embedding;
    EXECUTE 'CREATE INDEX idx_book_chunks_embedding ON book_chunks
        USING diskann (embedding)
        WITH (num_neighbors = 50, search_list_size = 100, max_alpha = 1.2)';
    RAISE NOTICE 'Book vector index created';
END;
$$ LANGUAGE plpgsql;
