"""Benchmark: rerank instruction-following with search_intent.

Compares retrieval quality across different search_intent values
using rerank-2.5 instruction-following on the same embeddings.

Usage:
    uv run python scripts/benchmark_models.py --project fluxomind-src
    uv run python scripts/benchmark_models.py --project fluxomind-src --chunks 500
"""

import argparse
import asyncio
import math
import time
from dataclasses import dataclass

import voyageai
from rich.console import Console
from rich.table import Table

from code_context.config import get_settings
from code_context.db.pool import DatabasePool
from code_context.retrieval.pipeline import RetrievalPipeline

console = Console()

# --- Config ---
EMBED_MODEL = "voyage-4-large"
QUERY_MODEL = "voyage-4-lite"
RERANK_MODEL = "rerank-2.5"
DIMENSIONS = 1024

# --- Test queries with expected best intent ---
TEST_CASES: list[dict] = [
    {"query": "authentication middleware that validates JWT tokens", "best_intent": "implementation"},
    {"query": "database connection pool setup and configuration", "best_intent": "implementation"},
    {"query": "error handling and retry logic for API calls", "best_intent": "debug"},
    {"query": "React component that renders a form with validation", "best_intent": "implementation"},
    {"query": "GraphQL resolver that fetches user data", "best_intent": "implementation"},
    {"query": "unit test for payment processing", "best_intent": "usage"},
    {"query": "type definitions for API request and response", "best_intent": "definition"},
    {"query": "caching strategy with Redis or in-memory cache", "best_intent": "implementation"},
    {"query": "file upload handling with multipart form data", "best_intent": "implementation"},
    {"query": "websocket connection management and event handling", "best_intent": "debug"},
]

INTENTS = [None, "implementation", "definition", "usage", "debug"]


@dataclass
class ChunkData:
    filepath: str
    chunk_text: str
    chunk_type: str
    symbol_name: str | None
    start_line: int


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


async def load_chunks(db: DatabasePool, project_id: str, limit: int) -> list[ChunkData]:
    async with db.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT c.filepath, c.chunk_text, c.chunk_type, c.symbol_name, c.start_line
            FROM code_chunks c
            JOIN code_files f ON c.filepath = f.filepath
            WHERE f.project_id = $1
              AND f.language NOT IN ('markdown', 'json')
              AND c.chunk_type != 'file'
            ORDER BY random()
            LIMIT $2
            """,
            project_id,
            limit,
        )
        return [
            ChunkData(
                filepath=r["filepath"],
                chunk_text=r["chunk_text"],
                chunk_type=r["chunk_type"],
                symbol_name=r["symbol_name"],
                start_line=r["start_line"],
            )
            for r in rows
        ]


def embed_batch(
    client: voyageai.Client,
    texts: list[str],
    model: str,
    input_type: str,
) -> list[list[float]]:
    all_embeddings: list[list[float]] = []
    batch_size = 128
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        result = client.embed(
            texts=batch, model=model, input_type=input_type,
            output_dimension=DIMENSIONS, output_dtype="float",
        )
        all_embeddings.extend(result.embeddings)
    return all_embeddings


def rerank_docs(
    client: voyageai.Client,
    query: str,
    documents: list[str],
    top_k: int,
) -> list[tuple[int, float]]:
    result = client.rerank(
        query=query, documents=documents,
        model=RERANK_MODEL, top_k=min(top_k, len(documents)),
    )
    return [(r.index, r.relevance_score) for r in result.results]


def top_k_by_similarity(
    query_emb: list[float],
    chunk_embeddings: list[list[float]],
    k: int,
) -> list[tuple[int, float]]:
    scores = [(i, cosine_similarity(query_emb, emb)) for i, emb in enumerate(chunk_embeddings)]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]


def fmt_chunk(c: ChunkData) -> str:
    fname = c.filepath.split("/")[-1]
    sym = c.symbol_name or "?"
    return f"{fname}:{c.start_line} ({c.chunk_type} {sym})"


def intent_label(intent: str | None) -> str:
    return intent or "default"


async def run_benchmark(
    project_id: str,
    chunk_limit: int,
    test_cases: list[dict],
    vector_top_k: int = 20,
    final_top_k: int = 5,
) -> None:
    settings = get_settings()
    client = voyageai.Client(api_key=settings.voyage_api_key)

    # --- Load chunks ---
    console.print(f"\n[bold]Loading chunks from project '{project_id}'...[/bold]")
    db = DatabasePool()
    await db.initialize()
    try:
        chunks = await load_chunks(db, project_id, chunk_limit)
    finally:
        await db.close()

    if not chunks:
        console.print(f"[red]No code chunks found for project '{project_id}'[/red]")
        return

    console.print(f"  Loaded [cyan]{len(chunks)}[/cyan] code chunks")
    texts = [c.chunk_text for c in chunks]

    # --- Embed once ---
    console.print(f"\n[bold]Embedding {len(chunks)} chunks with {EMBED_MODEL}...[/bold]")
    t0 = time.monotonic()
    embeddings = await asyncio.to_thread(embed_batch, client, texts, EMBED_MODEL, "document")
    console.print(f"  Done in {time.monotonic() - t0:.1f}s\n")

    # --- Run queries across all intents ---
    console.print(f"[bold]Running {len(test_cases)} queries × {len(INTENTS)} intents...[/bold]\n")

    all_results: list[dict] = []

    for qi, tc in enumerate(test_cases, 1):
        query = tc["query"]
        best_intent = tc["best_intent"]
        console.print(f"[dim]Q{qi}/{len(test_cases)}:[/dim] {query} [dim](expected: {best_intent})[/dim]")

        # Same query embedding for all intents
        q_emb = await asyncio.to_thread(
            lambda q=query: client.embed(
                [q], model=QUERY_MODEL, input_type="query", output_dimension=DIMENSIONS,
            ).embeddings[0]
        )

        # Same vector candidates for all intents
        top_candidates = top_k_by_similarity(q_emb, embeddings, vector_top_k)
        candidate_docs = [texts[idx] for idx, _ in top_candidates]

        # Rerank with each intent
        intent_scores: dict[str, dict] = {}
        for intent in INTENTS:
            rerank_query = RetrievalPipeline._build_rerank_query(query, intent)
            reranked = await asyncio.to_thread(
                rerank_docs, client, rerank_query, candidate_docs, final_top_k,
            )

            results = [
                {
                    "chunk": chunks[top_candidates[li][0]],
                    "rerank_score": score,
                }
                for li, score in reranked
            ]

            avg_score = sum(r["rerank_score"] for r in results) / len(results) if results else 0.0
            top1 = results[0] if results else None

            intent_scores[intent_label(intent)] = {
                "avg": avg_score,
                "top1_score": top1["rerank_score"] if top1 else 0.0,
                "top1_chunk": top1["chunk"] if top1 else None,
                "results": results,
            }

        # Determine which intent won
        winner = max(intent_scores.items(), key=lambda x: x[1]["avg"])
        winner_name = winner[0]
        expected_won = winner_name == best_intent

        all_results.append({
            "query": query,
            "best_intent": best_intent,
            "winner": winner_name,
            "expected_won": expected_won,
            "scores": intent_scores,
        })

        # Inline output
        parts = []
        for intent in INTENTS:
            label = intent_label(intent)
            s = intent_scores[label]
            marker = " ★" if label == winner_name else ""
            parts.append(f"{label}={s['avg']:.3f}{marker}")
        correct_marker = "[green]✓[/green]" if expected_won else "[red]✗[/red]"
        console.print(f"  {correct_marker} winner={winner_name} | {' | '.join(parts)}")

    # === SUMMARY ===
    console.print("\n")

    # Per-intent win count
    win_counts: dict[str, int] = {intent_label(i): 0 for i in INTENTS}
    correct_count = 0
    for r in all_results:
        win_counts[r["winner"]] += 1
        if r["expected_won"]:
            correct_count += 1

    summary = Table(title="Intent Benchmark Summary", show_header=True)
    summary.add_column("Intent", style="bold")
    summary.add_column("Wins", justify="center")
    summary.add_column("Avg Score (all queries)", justify="center")

    for intent in INTENTS:
        label = intent_label(intent)
        avg = sum(r["scores"][label]["avg"] for r in all_results) / len(all_results)
        summary.add_row(label, str(win_counts[label]), f"{avg:.4f}")

    console.print(summary)
    console.print(f"\nExpected intent won: [bold]{correct_count}/{len(all_results)}[/bold]\n")

    # Per-query detail
    detail = Table(title="Per-Query: Winner vs Expected", show_header=True, show_lines=True)
    detail.add_column("#", justify="right", width=3)
    detail.add_column("Query", max_width=40)
    detail.add_column("Expected", justify="center")
    detail.add_column("Winner", justify="center")
    detail.add_column("Default\navg", justify="center", style="dim")
    detail.add_column("Winner\navg", justify="center", style="green")
    detail.add_column("Δ vs default", justify="center")

    for i, r in enumerate(all_results, 1):
        default_avg = r["scores"]["default"]["avg"]
        winner_avg = r["scores"][r["winner"]]["avg"]
        delta = winner_avg - default_avg
        d_style = "green" if delta > 0 else ("red" if delta < 0 else "yellow")
        match = "[green]✓[/green]" if r["expected_won"] else "[red]✗[/red]"
        detail.add_row(
            str(i),
            r["query"][:40],
            r["best_intent"],
            f"{match} {r['winner']}",
            f"{default_avg:.3f}",
            f"{winner_avg:.3f}",
            f"[{d_style}]{delta:+.3f}[/{d_style}]",
        )

    console.print(detail)

    # Top-1 comparison: default vs expected intent
    console.print("\n[bold]Top-1: default vs matched intent[/bold]\n")
    for i, r in enumerate(all_results, 1):
        bi = r["best_intent"]
        default_top1 = r["scores"]["default"]["top1_chunk"]
        intent_top1 = r["scores"][bi]["top1_chunk"]
        console.print(f"[bold]Q{i}:[/bold] {r['query']}")
        if default_top1:
            console.print(f"  [dim]default[/dim]:        {fmt_chunk(default_top1)} rerank={r['scores']['default']['top1_score']:.3f}")
        if intent_top1:
            changed = default_top1 and (
                default_top1.filepath != intent_top1.filepath
                or default_top1.start_line != intent_top1.start_line
            )
            marker = " ← [bold green]NEW[/bold green]" if changed else ""
            console.print(f"  [green]{bi:14}[/green]: {fmt_chunk(intent_top1)} rerank={r['scores'][bi]['top1_score']:.3f}{marker}")
        console.print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark rerank search_intent")
    parser.add_argument("--project", required=True, help="Project ID to benchmark against")
    parser.add_argument("--chunks", type=int, default=500, help="Number of chunks to sample")
    parser.add_argument("--top-k", type=int, default=5, help="Final top-k results per query")

    args = parser.parse_args()

    asyncio.run(
        run_benchmark(
            project_id=args.project,
            chunk_limit=args.chunks,
            test_cases=TEST_CASES,
            final_top_k=args.top_k,
        )
    )


if __name__ == "__main__":
    main()
