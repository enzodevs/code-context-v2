# SPEC — AST-Aware Chunking + Hierarchical Retrieval (+ Hybrid Optional)

**Projeto:** `code-context-v2`
**Status:** Fases A, B, C, D concluídas. Fases E, F pendentes.
**Última atualização:** 2026-03-05
**Idioma da spec:** Português (termos técnicos em inglês)

---

## 1. Objetivo

Roadmap para elevar qualidade de retrieval sem inflar contexto inútil:

1. `AST-aware chunking` — eliminar split cego por token.
2. `hierarchical context assembly` — montagem de contexto por símbolo/arquivo com budget inteligente.
3. `observabilidade + benchmark` — decisões por evidência.
4. `cross-file context assembly` — enriquecer contexto com tipos/interfaces referenciados.
5. `hybrid retrieval (BM25 + Dense)` — opcional, somente se benchmark justificar.

---

## 2. Status por Fase

| Fase | Descrição | Status | Sessão |
|------|-----------|--------|--------|
| **A** | AST-aware chunking | **Concluída** | 2026-03-04 |
| **B** | Hierarchical context assembly | **Concluída** | 2026-03-04 |
| **C** | Observabilidade + benchmark | **Concluída** | 2026-03-05 |
| **D** | Cross-file context assembly | **Concluída** | 2026-03-05 |
| **E** | Parser multilíngue (Java/Rust) | Planejada | — |
| **F** | Hybrid retrieval (BM25 + Dense) | Opcional | — |

---

## 3. Resultados Medidos (cardify, 301 files, 3695 chunks)

### 3.1 Quality indicators

| Métrica | Pré-Phase A | Pós-Phase A+B+C | Delta |
|---------|:-----------:|:----------------:|:-----:|
| Signature coverage | 19.8% | **69.8%** | +50pp |
| Tiny chunks (<30 tok) | 236 | **0** | -236 |
| Oversized (>600 tok) | 3 | **0** | -3 |
| File chunk ratio (index) | 41.5% | **27.0%** | -14.5pp |
| File chunks nos resultados | 41.0% | **15.5%** | -25.5pp |
| Total chunks | 4622 | **3695** | -20% |

### 3.2 Retrieval benchmark (13 queries, E2E through MCP pipeline)

```
               hit@1    hit@3    hit@5      MRR    score       ms
architecture    1.00     1.00     1.00     1.00     0.68     600
debug           0.00     1.00     1.00     0.50     0.65     670
definition      0.50     1.00     1.00     0.75     0.71     590
identifier      1.00     1.00     1.00     1.00     0.76     686
performance     1.00     1.00     1.00     1.00     0.57     703
security        1.00     1.00     1.00     1.00     0.71     588
semantic        0.50     1.00     1.00     0.75     0.73     646
usage           0.00     0.00     1.00     0.25     0.73     614
─────────────────────────────────────────────────────────────
OVERALL         0.62     0.92     1.00     0.79     0.70     637
```

### 3.3 Per-file controls (Phase B) em ação

| Métrica | Antes (19 queries) | Depois (13 queries) |
|---------|:------------------:|:-------------------:|
| file_replaced total | 1 | **12** |
| symbol_capped total | 0 | **2** |
| results_returned avg | 3.2 | **5.5** |
| tokens_used avg | 1557 | **1816** |
| Latency p50 | 780ms | **613ms** |

### 3.4 Suporte por linguagem

| Linguagem | Symbol extraction | Per-file controls | Status |
|-----------|:-:|:-:|---|
| **Python** | Excelente | Pleno | Produção |
| **TypeScript/TSX** | Bom (70%) | Pleno | Produção |
| **JavaScript/JSX** | Bom (70%) | Pleno | Produção |
| **Go** | Bom | Parcial | Funcional |
| **SQL** | Bom | N/A | Funcional |
| **Markdown** | Bom (sections) | N/A | Funcional |
| **Java** | Fraco (~40%) | Limitado | Precisa de Phase E |
| **Rust** | Médio (~60%) | Limitado | Precisa de Phase E |

---

## 4. O que foi implementado

### Phase A — AST-Aware Chunking (concluída 2026-03-04)

**Arquivos:** `src/code_context/chunking/parser.py`, `tests/test_integration.py`

1. Tree-sitter AST parsing para 12 linguagens
2. Chunking hierárquico em 3 níveis: file → declaration → function/method
3. Split estrutural em statement boundaries (não por token)
4. Signature extraction em `context_metadata`
5. `min_tokens` re-filtra sub-chunks após split
6. Arrow function name extraction via AST ancestry walk (fix 2026-03-05)

### Phase B — Hierarchical Context Assembly (concluída 2026-03-04)

**Arquivos:** `src/code_context/retrieval/pipeline.py`, `src/code_context/config.py`, `tests/test_retrieval_controls.py`

1. Per-file token budget (`per_file_budget_ratio=0.45`)
2. Symbol prioritization: se um file tem symbol chunks, file chunk genérico é removido
3. `max_symbols_per_file=5` cap por arquivo
4. `max_file_chunks=2` cap global para file-level chunks
5. Test filter com safety net (relaxa se filtragem é excessiva)
6. 7 search intents com rerank instruction-following (rerank-2.5)
7. Relative threshold filtering (`rerank_relative_factor=0.75`, `score_floor=0.40`)

### Phase C — Observabilidade + Benchmark (concluída 2026-03-05)

**Arquivos:** `scripts/analyze_logs.py`, `scripts/benchmark_retrieval.py`, `cli/manage.py`, `cc2.sh`

1. `cc2 stats` — quality indicators (signature coverage, tiny/oversized, file ratio)
2. `cc2 analyze` — agregação do JSONL (outcomes, relevance, efficiency, latency, intents)
3. `cc2 benchmark PROJECT` — E2E benchmark com file_hit@k, MRR, rerank scores
4. Baseline save/compare (`--save NAME`, `--compare NAME`)
5. Menu gum integrado (Analyze search quality, Run benchmark)
6. `get_quality_stats()` no DatabasePool

### Melhorias de indexação (2026-03-05)

1. Ignore patterns: `.cxx`, `.expo`, `Pods` (removeu 152 files de build cache do index)
2. Arrow function name extraction: `_get_node_name()` sobe até `variable_declarator` na AST

### Phase D — Cross-File Context Assembly (concluída 2026-03-05)

**Arquivos:** `src/code_context/retrieval/pipeline.py`, `src/code_context/db/pool.py`, `src/code_context/config.py`, `tests/test_retrieval_controls.py`

1. Type reference extraction from `context_metadata` (signature + imports)
2. PascalCase type name detection with builtin exclusion set (`_BUILTIN_TYPE_NAMES`)
3. Named import parsing for JS/TS (`import { Foo } from '...'`) and Python (`from x import Foo`)
4. Frequency-based prioritization (types referenced by multiple chunks rank higher)
5. B-tree index lookup via `get_chunks_by_symbol_names()` — zero embedding/rerank calls
6. Budget-aware inclusion: only when `tokens_used < 70%` of max budget
7. Cap at `cross_file_max_chunks=3` per query
8. `[Referenced Type]` tag in formatted output for cross-file context chunks
9. Quality logging with `cross_file` section (refs_extracted, chunks_added, budget_skipped)
10. 6 unit tests covering happy path, budget guard, dedup, cap, disabled, no-project

---

## 5. Fases Pendentes

## Phase D — Cross-File Context Assembly (Implementada)

### D.1 Objetivo

Quando o pipeline retorna um chunk (ex: `function getBalance`), automaticamente incluir tipos, interfaces ou schemas que ele referencia — sem query adicional do usuário. Isso transforma resultados isolados em contexto auto-suficiente.

### D.2 Problema atual

O pipeline retorna chunks individuais. Se `getBalance` usa `UserStats` (definido em `schema.ts`), o usuário recebe o código da função mas não o tipo — precisa fazer outra query. Em 8000 tokens de budget, sobra espaço (uso médio é 1816 tokens = 23%).

### D.3 Estratégia proposta

1. **Na fase de formatting** (após rerank + dedup + per-file controls):
   - Para cada chunk retornado, extrair referências de tipos/interfaces do `context_metadata` (imports) e do texto do chunk (type annotations)
   - Fazer lookup no index: buscar chunks do tipo `declaration` cujo `symbol_name` match
   - Incluir como "context chunks" com score artificial baixo (não competem no ranking, aparecem ao final)

2. **Budget-aware**:
   - Só incluir cross-file chunks se `tokens_used < max_tokens * 0.7`
   - Cap de 2-3 cross-file chunks por query
   - Priorizar por frequência de referência (tipo usado por 3 chunks rankeados > tipo usado por 1)

3. **Sem impacto em latência**:
   - Lookup é por `symbol_name` exato no PostgreSQL (index B-tree)
   - Zero chamadas de embedding ou rerank extras

### D.4 Arquivos alvo

1. `src/code_context/retrieval/pipeline.py` — nova etapa entre per-file controls e formatting
2. `src/code_context/db/pool.py` — `get_chunks_by_symbol_names(names, project)` query
3. `tests/test_retrieval_controls.py` — testes de cross-file assembly

### D.5 Critérios de aceite

1. Query que retorna função usando tipo X inclui automaticamente a definição de X
2. Cross-file chunks não excedem budget
3. Sem regressão de latência p95 acima de 15%

---

## Phase E — Parser Multilíngue (Java/Rust)

### E.1 Objetivo

Elevar Java e Rust ao mesmo nível de symbol extraction de Python/TS, desbloqueando per-file controls para essas linguagens.

### E.2 Problemas identificados

**Java:**
- `_extract_functions` extrai apenas 1 método por classe em certos padrões
- Interfaces com method signatures não geram chunks individuais
- Construtores não são capturados separadamente
- Causa: tree-sitter Java tem `class_body` como nó intermediário; o visitor precisa recursion ajustada

**Rust:**
- `impl` blocks não têm `name` field — aparecem como `symbol_name=NULL`
- Métodos dentro de `impl` são extraídos mas sem qualificação (`new` em vez de `Config::new`)
- Traits extraem o nome mas não os methods individuais
- Métodos pequenos (<30 tokens) são filtrados pelo `min_tokens`

### E.3 Estratégia proposta

**Java:**
1. Ajustar `_extract_functions` para recursion em `class_body` nodes
2. Capturar todos os `method_declaration` dentro de classes (não só o primeiro)
3. Interface method declarations como chunks `declaration`
4. Qualificar: `ClassName.methodName`

**Rust:**
1. Para `impl_item`: extrair o tipo implementado (`impl Config` → nome `Config`)
2. Qualificar methods: `Config::new`, `Config::get`
3. Reduzir `min_tokens` para Rust (methods idiomáticos são curtos)
4. `trait_item` → extrair method signatures como `declaration` chunks

### E.4 Arquivos alvo

1. `src/code_context/chunking/parser.py` — ajustar `_extract_functions`, `_get_node_name`
2. `tests/test_integration.py` — casos de teste Java e Rust

### E.5 Critérios de aceite

1. Java: signature coverage >80% em projeto Spring Boot típico
2. Rust: signature coverage >70% em projeto com impl blocks
3. Sem regressão nos testes existentes de Python/TS

---

## Phase F — Hybrid Retrieval (Opcional)

### F.1 Quando iniciar

Somente se benchmark em projeto grande (>20k chunks) mostrar queda de recall em identifier queries com dense-only. Benchmark atual no cardify (3695 chunks) mostra 100% hit@5 para identifiers, indicando que embedding resolve para projetos deste porte.

### F.2 Estratégia

1. `tsvector` + GIN index no PostgreSQL sobre `chunk_text`
2. Busca keyword em paralelo com vector search
3. Fusão por RRF (Reciprocal Rank Fusion) ou weighted score
4. Rerank permanece como filtro final

### F.3 Feature flags

- `hybrid_search_enabled` (default: `false`)
- `hybrid_keyword_weight` (default: `0.3`)
- `hybrid_lexical_k` (default: `50`)

### F.4 Critério de promoção

Ativar por default apenas se ganho mensurável em MRR >5% em benchmark E2E.

---

## 6. Princípios de Design

1. **Precision-first:** contexto útil por token. Budget de 8000 tokens = ~4% de uma janela de 200k.
2. **AST-first:** chunk sempre em limite estrutural quando possível.
3. **Budget-aware:** `max_tokens` respeitado; usar subagentes com janela dedicada para análise profunda.
4. **Backward-compatible:** nenhuma quebra de interface MCP.
5. **Evidence-driven:** mudanças medidas por benchmark (`cc2 benchmark`) antes de promoção.

---

## 7. Ferramentas de Observabilidade

| Comando | Função |
|---------|--------|
| `cc2 stats` | Quality indicators globais (signature coverage, tiny/oversized, file ratio) |
| `cc2 analyze` | Agregação do JSONL de quality logging |
| `cc2 analyze --last N` | Últimas N queries |
| `cc2 benchmark PROJECT` | E2E benchmark com MRR, hit@k, rerank scores |
| `cc2 benchmark PROJECT --save NAME` | Salvar baseline |
| `cc2 benchmark PROJECT --compare NAME` | Comparar com baseline |

---

## 8. Configuração Atual

| Chave | Valor | Descrição |
|-------|-------|-----------|
| `ast_split_enabled` | `true` | Phase A ativa |
| `hierarchical_assembly_enabled` | `true` | Phase B ativa |
| `per_file_budget_ratio` | `0.45` | Max 45% do budget por arquivo |
| `max_symbols_per_file` | `5` | Cap de symbol chunks por arquivo |
| `rerank_relative_factor` | `0.75` | Threshold = top_score * factor |
| `rerank_score_floor` | `0.40` | Score mínimo absoluto |
| `result_max_tokens` | `8000` | Budget de tokens por query |
| `chunk_min_tokens` | `30` | Mínimo para reter chunk |
| `chunk_max_tokens` | `600` | Máximo antes de split |
| `cross_file_assembly_enabled` | `true` | Phase D ativa |
| `cross_file_max_chunks` | `3` | Max cross-file refs por query |
| `cross_file_budget_threshold` | `0.70` | Só inclui se uso < 70% do budget |

---

## 9. Baselines salvos

| Baseline | Data | MRR | hit@3 | hit@5 | avg_score | Notas |
|----------|------|:---:|:-----:|:-----:|:---------:|-------|
| `pre-parser-fix` | 2026-03-05 | 0.78 | 0.92 | 1.00 | 0.70 | Antes do fix de arrow function names |
| `post-parser-fix` | 2026-03-05 | 0.79 | 0.92 | 1.00 | 0.70 | Após fix + ignore cleanup |

---

## 10. Checklist Executivo

- [x] Phase A — AST-aware chunking
- [x] Phase B — Hierarchical context assembly
- [x] Phase C — Observabilidade + benchmark
- [x] Reindex completo e validação de stats
- [x] Benchmark com baseline salvo
- [x] Arrow function name extraction (TS/JS)
- [x] Ignore patterns para build cache (`.cxx`, `.expo`, `Pods`)
- [x] Phase D — Cross-file context assembly
- [ ] Phase E — Parser Java/Rust
- [ ] Phase F — Hybrid retrieval (se justificado por benchmark)

---

## 11. Riscos e Mitigações

1. **Risco:** cross-file assembly aumenta latência.
   **Mitigação:** lookup por symbol_name é B-tree index, sem embedding.

2. **Risco:** parser Java/Rust introduz regressão em Python/TS.
   **Mitigação:** testes isolados por linguagem, sem alteração das configs existentes.

3. **Risco:** reindex longo em bases grandes.
   **Mitigação:** execução por lotes + watchers pausados.

4. **Risco:** cross-file chunks consomem budget excessivo.
   **Mitigação:** cap de 2-3 chunks, só quando budget < 70% usado.
