-- Code Context RAG - Database Initialization
-- PostgreSQL 16 + pgvector + pgvectorscale

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;

-- Files tracking table
CREATE TABLE IF NOT EXISTS code_files (
    filepath TEXT PRIMARY KEY,
    file_hash TEXT NOT NULL,
    language TEXT NOT NULL,
    last_indexed TIMESTAMPTZ DEFAULT NOW(),
    loc INTEGER DEFAULT 0,
    project_root TEXT NOT NULL,
    project_id TEXT NOT NULL  -- Short identifier for the project (e.g., "my-app")
);

CREATE INDEX IF NOT EXISTS idx_files_project ON code_files(project_root);
CREATE INDEX IF NOT EXISTS idx_files_project_id ON code_files(project_id);
CREATE INDEX IF NOT EXISTS idx_files_language ON code_files(language);

-- Chunks with embeddings
CREATE TABLE IF NOT EXISTS code_chunks (
    id BIGSERIAL PRIMARY KEY,
    filepath TEXT NOT NULL REFERENCES code_files(filepath) ON DELETE CASCADE,
    chunk_hash TEXT NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector(1024),  -- voyage-4-large: 1024 dims
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    chunk_type TEXT NOT NULL,  -- function | class | method | module | import
    symbol_name TEXT,  -- function/class name if applicable
    context_metadata JSONB DEFAULT '{}',  -- module, imports, parent class, etc
    token_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for fast lookup
CREATE INDEX IF NOT EXISTS idx_chunks_filepath ON code_chunks(filepath);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON code_chunks(chunk_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_type ON code_chunks(chunk_type);
CREATE INDEX IF NOT EXISTS idx_chunks_symbol ON code_chunks(symbol_name) WHERE symbol_name IS NOT NULL;

-- StreamingDiskANN index for vector search (created after data is inserted)
-- This will be created by the application after initial indexing
-- CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON code_chunks
--     USING diskann (embedding)
--     WITH (num_neighbors = 50, search_list_size = 100, max_alpha = 1.2, num_bits_per_dimension = 2);

-- Function to create or recreate the vector index
CREATE OR REPLACE FUNCTION create_vector_index()
RETURNS void AS $$
BEGIN
    -- Ensure StreamingDiskANN index exists (incremental-friendly)
    EXECUTE 'CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON code_chunks
        USING diskann (embedding)
        WITH (num_neighbors = 50, search_list_size = 100, max_alpha = 1.2)';

    RAISE NOTICE 'Vector index ensured successfully';
END;
$$ LANGUAGE plpgsql;

-- Function for vector similarity search with pre-filtering
-- p_project_id filters by the short project identifier
-- p_languages filters by array of languages (for file_type='code' filtering)
CREATE OR REPLACE FUNCTION search_chunks(
    query_embedding vector(1024),
    p_filepath TEXT DEFAULT NULL,
    p_chunk_type TEXT DEFAULT NULL,
    p_language TEXT DEFAULT NULL,
    p_limit INTEGER DEFAULT 20,
    p_project_id TEXT DEFAULT NULL,
    p_languages TEXT[] DEFAULT NULL,
    p_directory TEXT DEFAULT NULL
)
RETURNS TABLE (
    chunk_id BIGINT,
    filepath TEXT,
    chunk_text TEXT,
    chunk_type TEXT,
    symbol_name TEXT,
    start_line INTEGER,
    end_line INTEGER,
    context_metadata JSONB,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        cc.id,
        cc.filepath,
        cc.chunk_text,
        cc.chunk_type,
        cc.symbol_name,
        cc.start_line,
        cc.end_line,
        cc.context_metadata,
        1 - (cc.embedding <=> query_embedding) AS similarity
    FROM code_chunks cc
    JOIN code_files cf ON cc.filepath = cf.filepath
    WHERE
        (p_filepath IS NULL OR cc.filepath = p_filepath)
        AND (p_chunk_type IS NULL OR cc.chunk_type = p_chunk_type)
        AND (p_language IS NULL OR cf.language = p_language)
        AND (p_project_id IS NULL OR cf.project_id = p_project_id)
        AND (p_languages IS NULL OR cf.language = ANY(p_languages))
        AND (p_directory IS NULL OR cc.filepath LIKE p_directory || '%')
        AND cc.embedding IS NOT NULL
    ORDER BY cc.embedding <=> query_embedding
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Stats view (includes hierarchical chunk types)
CREATE OR REPLACE VIEW index_stats AS
SELECT
    COUNT(DISTINCT cf.filepath) AS total_files,
    COUNT(cc.id) AS total_chunks,
    COUNT(DISTINCT cf.project_root) AS total_projects,
    ROUND(AVG(cc.token_count)) AS avg_tokens_per_chunk,
    ROUND(AVG(cf.loc)) AS avg_loc_per_file,
    cf.language,
    COUNT(*) FILTER (WHERE cc.chunk_type = 'file') AS files,
    COUNT(*) FILTER (WHERE cc.chunk_type = 'declaration') AS declarations,
    COUNT(*) FILTER (WHERE cc.chunk_type = 'function') AS functions,
    COUNT(*) FILTER (WHERE cc.chunk_type = 'class') AS classes,
    COUNT(*) FILTER (WHERE cc.chunk_type = 'method') AS methods
FROM code_files cf
LEFT JOIN code_chunks cc ON cf.filepath = cc.filepath
GROUP BY cf.language;

-- Cleanup function for removing orphaned data
CREATE OR REPLACE FUNCTION cleanup_orphaned_chunks()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM code_chunks
    WHERE filepath NOT IN (SELECT filepath FROM code_files);

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;
