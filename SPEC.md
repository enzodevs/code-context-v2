# Code Context v2 - Especificação Técnica Validada

## Stack SOTA (Janeiro 2026)

### Modelos Voyage AI

#### Indexação (One-time/Incremental)
```
Modelo: voyage-4-large
Input Type: document
Dimensões: 1024 (Matryoshka)
Quantização: float (pode usar int8 para storage)
Rate Limits: 3M TPM
Justificativa: MoE architecture, SOTA accuracy (RTEB #1), 40% mais barato que dense models
```

#### Queries Runtime (High-frequency)
```
Modelo: voyage-4-lite
Input Type: query
Dimensões: 1024 (shared embedding space com voyage-4-large)
Rate Limits: 16M TPM
Justificativa: Asymmetric retrieval pattern - mesmo embedding space, 5x throughput
```

#### Reranking
```
Modelo: rerank-2.5
Rate Limits: 4M TPM
Top-k input: 20 chunks
Top-k output: 8 chunks
Threshold: 0.70 relevance score
Justificativa: Maior precision para code (vs rerank-2.5-lite)
```

### Vector Database

```
Stack: PostgreSQL 16 + pgvector 0.8+ + pgvectorscale
Porta: 54329 (evita conflitos)
Índice: StreamingDiskANN
Configuração:
  - num_neighbors: 50
  - search_list_size: 100
  - max_alpha: 1.2

Performance: 471 QPS @ 99% recall (11.4x vs Qdrant)
Latência: <100ms p99
```

### Linguagens Suportadas

```
TypeScript: .ts, .tsx (tree-sitter-typescript)
JavaScript: .js, .jsx, .mjs, .cjs (tree-sitter-javascript)
Python: .py, .pyi (tree-sitter-python)
Java: .java (tree-sitter-java)
```

## Arquitetura

### MCP Server (FastMCP)

```
Tools expostas: 2 (minimal context window usage)

1. search_codebase(query, top_k, language?, chunk_type?)
   - Semantic search across entire codebase
   - Vector search + reranking automático
   - Context formatting otimizado

2. search_by_file(filepath, query, top_k?)
   - Busca escopo limitado a um arquivo
   - Útil para refactoring local
```

### CLI (gum TUI) - Separado do MCP

```bash
code-context-cli index <path>      # Index project
code-context-cli reindex <path>    # Force reindex
code-context-cli stats             # Show statistics
code-context-cli watch <path>      # Real-time indexing
code-context-cli prune             # Remove orphaned data
```

## Context Engineering Pipeline

### 1. Chunking Strategy

```
Método: AST-based (tree-sitter)
Unidades: Funções/classes/methods completos
Tamanho: 10-800 tokens/chunk
Contextual Enrichment:
  - File path absoluto
  - Symbol name (function/class/method)
  - Imports relevantes (top 5)
  - Parent class (para methods)
```

### 2. Incremental Ingestion

```
Hash: BLAKE3 do file content
Update Logic:
  - Hash match → skip
  - Hash diff → DELETE old chunks + INSERT new (transação atômica)
  - Novo arquivo → INSERT chunks

Batch Size: 128 textos/request (Voyage limit)
```

### 3. Retrieval Pipeline

```
Stage 1 - Vector Search:
  - Query embedding: voyage-4-lite (input_type=query)
  - Top-k: 20 chunks
  - Filtros SQL: filepath, chunk_type, language (pre-filter)
  - Método: Cosine similarity (pgvector <=> operator)

Stage 2 - Reranking:
  - Input: 20 chunks from Stage 1
  - Output: 8 chunks (threshold: relevance_score > 0.70)
  - Modelo: rerank-2.5

Stage 3 - Deduplication:
  - Remove chunks com >95% text similarity (Jaccard)
  - Mantém chunk com maior score

Stage 4 - Context Assembly:
  - Ordenação: Relevance score DESC
  - Posicionamento: Mais relevantes no INÍCIO (primacy effect)
  - Formato: Markdown com filepath, lines, relevance score
```

### 4. Response Format

```markdown
## Result 1 (relevance: 0.89)
```
File: /path/to/file.ts:10-25 | Symbol: validateEmail | Type: function
---
export function validateEmail(email: string): boolean {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+/.test(email);
}
```

## Result 2 (relevance: 0.85)
...
```

## Database Schema

```sql
-- Files tracking
CREATE TABLE code_files (
    filepath TEXT PRIMARY KEY,
    file_hash TEXT NOT NULL,
    language TEXT NOT NULL,
    project_root TEXT NOT NULL,
    last_indexed TIMESTAMPTZ,
    loc INTEGER
);

-- Chunks with embeddings
CREATE TABLE code_chunks (
    id BIGSERIAL PRIMARY KEY,
    filepath TEXT REFERENCES code_files ON DELETE CASCADE,
    chunk_hash TEXT NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector(1024),
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    chunk_type TEXT NOT NULL,  -- function|class|method|module
    symbol_name TEXT,
    context_metadata JSONB,
    token_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- StreamingDiskANN index
CREATE INDEX idx_chunks_embedding ON code_chunks
    USING diskann (embedding)
    WITH (num_neighbors = 50, search_list_size = 100, max_alpha = 1.2);
```

## Performance Targets

```
Indexação:
  - 1000 arquivos: ~5-10 min (initial)
  - Single file update: <2s

Query Latency:
  - Vector search: <50ms
  - Reranking: <100ms
  - Total MCP response: <200ms

Storage (100k chunks):
  - 1024 dims float32: ~400MB vetores
  - Total com metadata: ~600MB

Accuracy:
  - Top-8 recall: >95%
  - Relevance threshold: 0.70+
```

## Claude Code Configuration

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
        "VOYAGE_API_KEY": "pa-...",
        "DATABASE_URL": "postgresql://coderag:password@localhost:54329/coderag"
      }
    }
  }
}
```

## Context Engineering Rules (Runtime)

1. Nunca envie >90% da context window
2. Priorize chunks com relevance >0.70
3. Deduplique chunks semanticamente similares (Jaccard >0.95)
4. Posicione chunks críticos no INÍCIO do contexto (primacy effect)
5. Use markdown com metadata explícito (file, lines, relevance)
6. Limite a 8 chunks por query (quality over quantity)
7. Inclua context headers (imports, parent class) em cada chunk
