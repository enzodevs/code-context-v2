<p align="center">
  <img src="assets/logo.png" alt="Code Context v2" width="600">
</p>

<p align="center">
  Semantic code search as an MCP server. Index your codebases, search with natural language, get precise results.
</p>

Built for [Claude Code](https://docs.anthropic.com/en/docs/claude-code) but works with any MCP-compatible client.

## Why

LLMs work better with the right context. Grep finds text; this finds **meaning**. Ask for "authentication middleware" and get the actual auth logic, not every file that mentions "auth".

**How it works:**
1. **Index** your codebase with tree-sitter AST parsing (functions, classes, methods — not arbitrary line splits)
2. **Embed** chunks with Voyage AI (`voyage-4-large` for documents, `voyage-4-lite` for queries — same embedding space, asymmetric retrieval)
3. **Search** with vector similarity + `rerank-2.5` for precision
4. **Serve** results via MCP protocol — Claude Code calls it automatically

## Architecture

```
Claude Code ──MCP──▶ FastMCP Server
                        │
                   search_codebase()
                   search_by_file()
                   list_projects()
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
         Voyage AI           PostgreSQL 16
      voyage-4-lite          + pgvector
       (query embed)         + pgvectorscale
      rerank-2.5             (StreamingDiskANN)
       (reranking)
```

**Retrieval pipeline:**
1. Embed query with `voyage-4-lite` (fast, shared space with indexed docs)
2. Vector search: top-50 candidates via cosine similarity (pgvector `<=>`)
3. Rerank: `rerank-2.5` narrows to top-8 with relevance threshold
4. Dedup: Jaccard similarity >95% removal
5. Return: Markdown-formatted chunks with file paths, line numbers, relevance scores

## Quick Start

### Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [Docker](https://docs.docker.com/get-docker/) (for PostgreSQL)
- [Voyage AI API key](https://dash.voyageai.com/) (free tier available)

### 1. Clone and configure

```bash
git clone https://github.com/YOUR_USER/code-context-v2.git
cd code-context-v2

cp .env.example .env
# Edit .env — set VOYAGE_API_KEY and POSTGRES_PASSWORD
```

### 2. Start PostgreSQL

```bash
docker compose up -d
```

This runs PostgreSQL 16 with pgvector + pgvectorscale (TimescaleDB image) on port `54329`.

### 3. Install dependencies

```bash
uv sync
```

### 4. Index a project

```bash
uv run code-context-cli index /path/to/your/project
```

### 5. Connect to Claude Code

Add to `~/.claude/mcp.json` (global) or `.mcp.json` (per-project):

```json
{
  "mcpServers": {
    "code-context": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/code-context-v2",
        "run",
        "code-context"
      ],
      "env": {
        "VOYAGE_API_KEY": "${VOYAGE_API_KEY}",
        "DATABASE_URL": "postgresql://coderag:your_password@localhost:54329/coderag"
      }
    }
  }
}
```

Restart Claude Code. It will automatically discover the `search_codebase`, `search_by_file`, and `list_projects` tools.

## MCP Tools

| Tool | Purpose |
|------|---------|
| `list_projects` | List all indexed projects (call first to get project IDs) |
| `search_codebase(query, project)` | Semantic search across entire codebase |
| `search_by_file(filepath, query, project)` | Search within a specific file |
| `list_books` | List indexed books (optional literature feature) |
| `search_literature(query, book?)` | Search indexed technical books |

## CLI

```bash
# Index a project (auto-generates ID from folder name)
uv run code-context-cli index /path/to/project

# Index with custom ID
uv run code-context-cli index /path/to/project --id my-project

# Check what changed (dry-run)
uv run code-context-cli check my-project

# Sync only changed files
uv run code-context-cli sync my-project

# Force full reindex
uv run code-context-cli reindex /path/to/project

# Show statistics
uv run code-context-cli stats

# Watch for changes (background daemon)
uv run code-context-cli watch /path/to/project

# Remove orphaned data
uv run code-context-cli prune
```

There's also `cc2.sh` — a bash wrapper with a gum-based TUI for interactive use.

## Supported Languages

| Language | Extensions | Parser |
|----------|-----------|--------|
| TypeScript | `.ts`, `.tsx` | tree-sitter-typescript |
| JavaScript | `.js`, `.jsx`, `.mjs`, `.cjs` | tree-sitter-javascript |
| Python | `.py`, `.pyi` | tree-sitter-python |
| Java | `.java` | tree-sitter-java |

Adding a new language requires a tree-sitter grammar and chunk type mappings in `src/code_context/chunking/parser.py`.

## Configuration

All settings via environment variables (or `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | — | PostgreSQL connection string (required) |
| `VOYAGE_API_KEY` | — | Voyage AI API key (required) |
| `EMBEDDING_MODEL_INDEX` | `voyage-4-large` | Embedding model for indexing |
| `EMBEDDING_MODEL_QUERY` | `voyage-4-lite` | Embedding model for queries |
| `RERANK_MODEL` | `rerank-2.5` | Reranking model |
| `RERANK_THRESHOLD` | `0.65` | Minimum relevance score after reranking |
| `RESULT_MAX_TOKENS` | `8000` | Token budget for results |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

See `src/code_context/config.py` for all available settings.

## Performance

- **Vector search:** <50ms
- **Reranking:** <100ms
- **Total MCP response:** <200ms
- **Initial indexing:** ~5-10 min for 1000 files
- **Incremental sync:** <2s per changed file
- **Storage:** ~100MB per 100k chunks

## How Indexing Works

1. Walk the project tree (skips `node_modules`, `.git`, `dist`, etc.)
2. Hash each file with BLAKE3 — skip unchanged files
3. Parse with tree-sitter into semantic chunks (functions, classes, methods)
4. Small files (<200 lines) are kept as single chunks to avoid fragmentation
5. Embed chunks with `voyage-4-large` in batches
6. Store in PostgreSQL with pgvector embeddings
7. All operations are atomic — Ctrl+C won't corrupt the index

## License

MIT
