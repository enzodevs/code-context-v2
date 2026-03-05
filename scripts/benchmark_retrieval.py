"""E2E retrieval benchmark through the real MCP pipeline.

Measures retrieval quality with curated queries and golden expectations.
Differs from benchmark_models.py which isolates rerank intents — this one
measures what the user actually receives.

Usage:
    uv run python -m scripts.benchmark_retrieval cardify
    uv run python -m scripts.benchmark_retrieval cardify --save baseline-v1
    uv run python -m scripts.benchmark_retrieval cardify --compare baseline-v1
"""

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from code_context.db.pool import DatabasePool
from code_context.embedding.voyage import VoyageClient
from code_context.retrieval.pipeline import RetrievalPipeline

BASELINES_DIR = Path(__file__).parent / "baselines"

# Benchmark queries organized by category.
# Each query specifies expected files and optionally expected symbols.
# Add/modify queries as your projects evolve.
BENCHMARK_QUERIES: dict[str, list[dict]] = {
    "cardify": [
        # --- Identifier (exact match critical) ---
        {
            "query": "getAuthUserId function",
            "category": "identifier",
            "expect_files": ["convex/lib/auth.ts", "convex/notifications.ts", "convex/profiles.ts"],
            "expect_symbols": ["getCurrentUserId", "getAuthUserId"],
        },
        {
            "query": "scheduleCard FSRS function",
            "category": "identifier",
            "expect_files": ["convex/lib/fsrs.ts"],
            "expect_symbols": ["scheduleCard"],
        },
        # --- Semantic (understanding intent) ---
        {
            "query": "how flashcard study sessions track progress and XP",
            "category": "semantic",
            "expect_files": ["convex/study.ts", "convex/gamification.ts"],
        },
        {
            "query": "stripe webhook handling for subscription events",
            "category": "semantic",
            "expect_files": ["convex/subscriptions.ts", "convex/http.ts"],
        },
        # --- Debug/error path ---
        {
            "query": "error handling when subscription check fails",
            "category": "debug",
            "search_intent": "debug",
            "expect_files": ["convex/subscriptions.ts"],
        },
        {
            "query": "validation and error cases in card creation",
            "category": "debug",
            "search_intent": "debug",
            "expect_files": ["convex/cards.ts", "convex/importSources.ts"],
        },
        # --- Architecture ---
        {
            "query": "economy system hearts keys credits store purchases",
            "category": "architecture",
            "search_intent": "architecture",
            "expect_files": ["convex/economy.ts"],
        },
        {
            "query": "achievement and gamification system logic",
            "category": "architecture",
            "search_intent": "architecture",
            "expect_files": ["convex/gamification.ts"],
        },
        # --- Definition ---
        {
            "query": "user profile schema and types",
            "category": "definition",
            "search_intent": "definition",
            "expect_files": ["convex/schema.ts", "convex/profiles.ts"],
        },
        {
            "query": "deck and card data model definitions",
            "category": "definition",
            "search_intent": "definition",
            "expect_files": ["convex/schema.ts", "convex/cards.ts", "convex/decks.ts"],
            "expect_symbols": ["cardValidator", "deckValidator", "defineSchema_part1"],
        },
        # --- Usage ---
        {
            "query": "where isPremium is checked before allowing access",
            "category": "usage",
            "search_intent": "usage",
            "expect_files": ["convex/lib/premium.ts", "convex/subscriptions.ts"],
            "expect_symbols": ["isPremium"],
        },
        # --- Performance ---
        {
            "query": "database queries that could be slow with many cards",
            "category": "performance",
            "search_intent": "performance",
            "expect_files": ["convex/cards.ts", "convex/study.ts", "convex/decks.ts"],
        },
        # --- Security ---
        {
            "query": "authentication checks and userId validation before mutations",
            "category": "security",
            "search_intent": "security",
            "expect_files": ["convex/lib/auth.ts", "convex/study.ts", "convex/cards.ts", "convex/profiles.ts"],
        },
    ],
}


@dataclass
class QueryResult:
    """Result for a single benchmark query."""
    query: str
    category: str
    search_intent: str | None
    expect_files: list[str]
    expect_symbols: list[str]
    returned_files: list[str]
    returned_symbols: list[str]
    scores: list[float]
    duration_ms: int
    file_hit_at: dict[int, bool] = field(default_factory=dict)
    symbol_hit_at: dict[int, bool] = field(default_factory=dict)
    mrr: float = 0.0


def _file_matches(returned: str, expected: str) -> bool:
    """Check if returned filepath ends with expected relative path."""
    return returned.endswith("/" + expected) or returned.endswith("\\" + expected)


def _compute_hits(result: QueryResult) -> None:
    """Compute file_hit@k, symbol_hit@k, and MRR for a query result."""
    for k in (1, 3, 5):
        top_k_files = result.returned_files[:k]
        result.file_hit_at[k] = any(
            _file_matches(rf, ef)
            for rf in top_k_files
            for ef in result.expect_files
        ) if result.expect_files else True

        top_k_symbols = result.returned_symbols[:k]
        result.symbol_hit_at[k] = any(
            es in (rs or "") for rs in top_k_symbols for es in result.expect_symbols
        ) if result.expect_symbols else True

    # MRR: reciprocal rank of first golden file hit
    if result.expect_files:
        for rank, rf in enumerate(result.returned_files, 1):
            if any(_file_matches(rf, ef) for ef in result.expect_files):
                result.mrr = 1.0 / rank
                break
    else:
        result.mrr = 1.0  # No expectation = skip


async def run_benchmark(
    project_id: str,
    pipeline: RetrievalPipeline,
) -> list[QueryResult]:
    """Run all benchmark queries for a project."""
    queries = BENCHMARK_QUERIES.get(project_id, [])
    if not queries:
        print(f"No benchmark queries defined for project '{project_id}'.")
        print(f"Available projects: {', '.join(BENCHMARK_QUERIES.keys())}")
        return []

    results: list[QueryResult] = []
    for i, case in enumerate(queries, 1):
        import time
        t0 = time.monotonic()

        search_results = await pipeline.search(
            query=case["query"],
            project=project_id,
            search_intent=case.get("search_intent"),
            file_type="code",
            _tool_name="benchmark",
        )

        duration_ms = int((time.monotonic() - t0) * 1000)

        qr = QueryResult(
            query=case["query"],
            category=case["category"],
            search_intent=case.get("search_intent"),
            expect_files=case.get("expect_files", []),
            expect_symbols=case.get("expect_symbols", []),
            returned_files=[r.filepath for r in search_results],
            returned_symbols=[r.symbol_name or "" for r in search_results],
            scores=[r.relevance_score for r in search_results],
            duration_ms=duration_ms,
        )
        _compute_hits(qr)
        results.append(qr)

        status = "HIT" if qr.file_hit_at.get(3, False) else "MISS"
        print(f"  [{i}/{len(queries)}] {status}  {case['query'][:60]}  ({duration_ms}ms)")

    return results


def _aggregate_by_category(results: list[QueryResult]) -> dict[str, dict]:
    """Group results by category and compute averages."""
    categories: dict[str, list[QueryResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    aggregated: dict[str, dict] = {}
    for cat, cat_results in categories.items():
        n = len(cat_results)
        has_expectations = [r for r in cat_results if r.expect_files]
        n_with_expect = len(has_expectations) or 1
        aggregated[cat] = {
            "n": n,
            "file_hit@1": sum(r.file_hit_at.get(1, False) for r in has_expectations) / n_with_expect,
            "file_hit@3": sum(r.file_hit_at.get(3, False) for r in has_expectations) / n_with_expect,
            "file_hit@5": sum(r.file_hit_at.get(5, False) for r in has_expectations) / n_with_expect,
            "MRR": sum(r.mrr for r in has_expectations) / n_with_expect,
            "avg_score": (
                sum(r.scores[0] for r in cat_results if r.scores) /
                sum(1 for r in cat_results if r.scores)
            ) if any(r.scores for r in cat_results) else 0,
            "avg_latency_ms": sum(r.duration_ms for r in cat_results) / n,
        }
    return aggregated


def print_table(project_id: str, results: list[QueryResult], comparison: dict | None = None) -> dict:
    """Print results table and return serializable summary."""
    aggregated = _aggregate_by_category(results)

    print(f"\n=== Retrieval Benchmark ({project_id}, {len(results)} queries) ===\n")
    print(f"{'':14s} {'hit@1':>8s} {'hit@3':>8s} {'hit@5':>8s} {'MRR':>8s} {'score':>8s} {'ms':>8s}")
    print("─" * 66)

    for cat, m in sorted(aggregated.items()):
        line = (
            f"{cat:14s} {m['file_hit@1']:8.2f} {m['file_hit@3']:8.2f} "
            f"{m['file_hit@5']:8.2f} {m['MRR']:8.2f} {m['avg_score']:8.2f} "
            f"{m['avg_latency_ms']:7.0f}"
        )

        if comparison:
            old = comparison.get("by_category", {}).get(cat)
            if old:
                delta_mrr = m["MRR"] - old.get("MRR", 0)
                delta_score = m["avg_score"] - old.get("avg_score", 0)
                sign_mrr = "+" if delta_mrr >= 0 else ""
                sign_score = "+" if delta_score >= 0 else ""
                line += f"  MRR:{sign_mrr}{delta_mrr:.2f} score:{sign_score}{delta_score:.2f}"

        print(line)

    # Overall
    all_with_expect = [r for r in results if r.expect_files]
    n_we = len(all_with_expect) or 1
    overall = {
        "file_hit@1": sum(r.file_hit_at.get(1, False) for r in all_with_expect) / n_we,
        "file_hit@3": sum(r.file_hit_at.get(3, False) for r in all_with_expect) / n_we,
        "file_hit@5": sum(r.file_hit_at.get(5, False) for r in all_with_expect) / n_we,
        "MRR": sum(r.mrr for r in all_with_expect) / n_we,
        "avg_score": (
            sum(r.scores[0] for r in results if r.scores) /
            sum(1 for r in results if r.scores)
        ) if any(r.scores for r in results) else 0,
        "avg_latency_ms": sum(r.duration_ms for r in results) / len(results),
    }

    print("─" * 66)
    print(
        f"{'OVERALL':14s} {overall['file_hit@1']:8.2f} {overall['file_hit@3']:8.2f} "
        f"{overall['file_hit@5']:8.2f} {overall['MRR']:8.2f} {overall['avg_score']:8.2f} "
        f"{overall['avg_latency_ms']:7.0f}"
    )

    if comparison:
        old_overall = comparison.get("overall", {})
        if old_overall:
            d_mrr = overall["MRR"] - old_overall.get("MRR", 0)
            d_score = overall["avg_score"] - old_overall.get("avg_score", 0)
            s_mrr = "+" if d_mrr >= 0 else ""
            s_score = "+" if d_score >= 0 else ""
            print(f"\n  vs baseline: MRR {s_mrr}{d_mrr:.3f}  score {s_score}{d_score:.3f}")

    print()

    return {
        "project": project_id,
        "n_queries": len(results),
        "overall": overall,
        "by_category": aggregated,
    }


def save_baseline(summary: dict, name: str) -> Path:
    """Save benchmark results as a baseline."""
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    path = BASELINES_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    return path


def load_baseline(name: str) -> dict | None:
    """Load a saved baseline."""
    path = BASELINES_DIR / f"{name}.json"
    if not path.exists():
        print(f"Baseline not found: {path}")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


async def main_async(project_id: str, save_name: str | None, compare_name: str | None) -> None:
    db = DatabasePool()
    await db.initialize()

    try:
        voyage = VoyageClient()
        pipeline = RetrievalPipeline(db, voyage)

        print(f"\nRunning benchmark for '{project_id}'...\n")
        results = await run_benchmark(project_id, pipeline)

        if not results:
            return

        comparison = load_baseline(compare_name) if compare_name else None
        summary = print_table(project_id, results, comparison)

        if save_name:
            path = save_baseline(summary, save_name)
            print(f"Baseline saved: {path}")

    finally:
        await db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="E2E retrieval benchmark")
    parser.add_argument("project", help="Project ID to benchmark")
    parser.add_argument("--save", metavar="NAME", help="Save results as named baseline")
    parser.add_argument("--compare", metavar="NAME", help="Compare against a saved baseline")
    args = parser.parse_args()

    asyncio.run(main_async(args.project, args.save, args.compare))


if __name__ == "__main__":
    main()
