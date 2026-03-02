# AGENTS — Code Context v2

Este arquivo define diretrizes globais para agentes que usam ou evoluem este repositório.

## Objetivo de Produto

Maximizar relevância de contexto para agentes de código com o menor ruído possível.

Princípios:
- Priorizar precisão contextual sobre volume bruto.
- Evitar difusão de atenção do agente cliente.
- Rastrear mudanças de qualidade com logs e benchmarks reproduzíveis.

## Search Intents (Rerank)

Intents suportados:
- `implementation`
- `definition`
- `usage`
- `debug`
- `security`
- `performance`
- `architecture`

Default:
- Quando `search_intent` não é informado, usar `implementation`.

## Guia de Uso dos Intents

- `implementation`: quando a meta é alterar/entregar comportamento concreto.
- `definition`: quando a meta é encontrar contratos, tipos, schemas e declarações.
- `usage`: quando a meta é encontrar call sites e integração entre módulos.
- `debug`: quando a meta é investigar falhas, retries, timeouts, validações e logs.
- `security`: quando a meta é auth/authz, sanitização, segredos, hardening.
- `performance`: quando a meta é latência, throughput, caching, batching e hotspots.
- `architecture`: quando a meta é fronteiras entre módulos e fluxo cross-layer.

## Regras de Qualidade de Contexto

- Preferir chunks de símbolo (`function/method/class/declaration`) a chunks genéricos de arquivo quando ambos respondem à query.
- Penalizar código de teste (`test/spec/fixture/mock`) quando a query não pedir explicitamente testes.
- Evitar snippets de baixa densidade semântica para a pergunta feita.
- Para reduzir custo de contexto, priorizar uso de `max_tokens`, `include_tests=false` e `max_file_chunks` baixo.

## Mudanças de Prompt

- Toda mudança de instrução de rerank deve atualizar a versão de prompt no pipeline (`RERANK_PROMPT_VERSION`).
- Comparações devem ser feitas com o mesmo conjunto de queries e parâmetros de filtro.
- Usar arquivos JSONL separados por experimento (`/tmp/cc2-search-*.jsonl`).

## Protocolo de Avaliação

Checklist mínimo por rodada:
1. Reexecutar baseline reproduzível (mesmo TSV de queries).
2. Comparar:
   - média de resultados retornados;
   - contagem de buscas com `< max_results`;
   - latência média;
   - taxa de ruído (ex.: presença de arquivos de teste);
   - regressões por query.
3. Só promover configuração se não houver regressões relevantes.

## Documentação Obrigatória

Ao alterar intents/rerank:
- Atualizar `README.md` (guia de intents).
- Atualizar docstrings das tools em `src/code_context/server.py`.
- Atualizar benchmark em `scripts/benchmark_models.py` quando a taxonomy mudar.
