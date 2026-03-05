"""Analyze search quality JSONL logs.

Reads the SEARCH_LOG_PATH file and aggregates retrieval quality metrics.

Usage:
    uv run python -m scripts.analyze_logs
    uv run python -m scripts.analyze_logs --last 50
    uv run python -m scripts.analyze_logs --since 2025-01-01
"""

import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


def _load_entries(log_path: str, *, last: int | None = None, since: str | None = None) -> list[dict]:
    """Load and filter JSONL entries."""
    path = Path(log_path)
    if not path.exists():
        return []

    entries: list[dict] = []
    since_dt = datetime.fromisoformat(since) if since else None

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if since_dt:
                ts = entry.get("ts", "")
                try:
                    entry_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if entry_dt < since_dt.replace(tzinfo=entry_dt.tzinfo):
                        continue
                except (ValueError, TypeError):
                    continue

            entries.append(entry)

    if last is not None:
        entries = entries[-last:]

    return entries


def _percentile(data: list[float] | list[int], p: float) -> float:
    """Calculate percentile from sorted data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def analyze(entries: list[dict]) -> None:
    """Aggregate and print quality report."""
    n = len(entries)
    if n == 0:
        print("No log entries found.")
        return

    # Outcomes
    outcomes: Counter[str] = Counter()
    # Relevance
    top1_scores: list[float] = []
    top5_avgs: list[float] = []
    # Efficiency
    tokens_used: list[int] = []
    results_returned: list[int] = []
    file_chunk_counts: list[int] = []
    total_chunk_counts: list[int] = []
    # Per-file controls
    file_replaced_total = 0
    symbol_capped_total = 0
    budget_skipped_total = 0
    # Latency
    durations: list[int] = []
    # Intents
    intents: Counter[str] = Counter()

    for e in entries:
        outcome = e.get("outcome", "unknown")
        outcomes[outcome] += 1

        results = e.get("results", [])
        budget = e.get("budget", {})
        retrieval = e.get("retrieval", {})
        filters = e.get("filters", {})

        # Relevance
        if results:
            scores = [r.get("score", 0) for r in results]
            top1_scores.append(scores[0] if scores else 0)
            top5_avgs.append(statistics.mean(scores[:5]) if scores else 0)

        # Efficiency
        tokens_used.append(budget.get("used_tokens", 0))
        results_returned.append(budget.get("returned_results", 0))

        # File chunk tracking
        fc = sum(1 for r in results if r.get("type") == "file")
        file_chunk_counts.append(fc)
        total_chunk_counts.append(len(results))

        # Per-file controls
        file_replaced_total += retrieval.get("per_file_file_replaced", 0)
        symbol_capped_total += retrieval.get("per_file_symbol_capped", 0)
        budget_skipped_total += retrieval.get("per_file_budget_skipped", 0)

        # Latency
        if "duration_ms" in e:
            durations.append(e["duration_ms"])

        # Intent
        intent = filters.get("search_intent", "implementation") or "implementation"
        intents[intent] += 1

    # Print report
    print(f"\n=== Search Quality Report (last {n} queries) ===\n")

    # Outcomes
    print("Outcomes:")
    for outcome in ["ok", "no_candidates", "below_similarity_threshold", "empty_after_budget"]:
        count = outcomes.get(outcome, 0)
        pct = count / n * 100
        print(f"  {outcome + ':':25s} {count:4d} ({pct:.1f}%)")

    # Relevance
    if top1_scores:
        print(f"\nRelevance:")
        print(
            f"  top1_relevance:   avg={statistics.mean(top1_scores):.2f}  "
            f"p50={_percentile(top1_scores, 50):.2f}  "
            f"p25={_percentile(top1_scores, 25):.2f}"
        )
        if top5_avgs:
            print(
                f"  top5_avg:         avg={statistics.mean(top5_avgs):.2f}  "
                f"p50={_percentile(top5_avgs, 50):.2f}"
            )

    # Efficiency
    if tokens_used:
        print(f"\nEfficiency:")
        print(
            f"  tokens_used:      avg={statistics.mean(tokens_used):.0f}  "
            f"p50={_percentile(tokens_used, 50):.0f}  "
            f"max={max(tokens_used)}"
        )
        print(
            f"  results_returned: avg={statistics.mean(results_returned):.1f}  "
            f"p50={_percentile(results_returned, 50):.0f}"
        )
        total_fc = sum(file_chunk_counts)
        total_tc = sum(total_chunk_counts)
        fc_pct = (total_fc / total_tc * 100) if total_tc else 0
        print(f"  %file_chunks:     {fc_pct:.1f}%")

    # Per-file controls
    print(f"\nPer-file controls:")
    print(f"  file_replaced:    avg={file_replaced_total / n:.1f}   total={file_replaced_total}")
    print(f"  symbol_capped:    avg={symbol_capped_total / n:.1f}   total={symbol_capped_total}")
    print(f"  budget_skipped:   avg={budget_skipped_total / n:.1f}   total={budget_skipped_total}")

    # Latency
    if durations:
        print(f"\nLatency:")
        print(
            f"  p50={_percentile(durations, 50):.0f}ms  "
            f"p95={_percentile(durations, 95):.0f}ms  "
            f"max={max(durations)}ms"
        )

    # Intent distribution
    if intents:
        print(f"\nIntent distribution:")
        for intent, count in intents.most_common():
            pct = count / n * 100
            print(f"  {intent + ':':18s} {count:4d} ({pct:.1f}%)")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze search quality JSONL logs")
    parser.add_argument("--last", type=int, help="Analyze only the last N queries")
    parser.add_argument("--since", help="Analyze queries since YYYY-MM-DD")
    parser.add_argument("--log-path", help="Override log path (default: from .env)")
    args = parser.parse_args()

    log_path = args.log_path
    if not log_path:
        # Read from config
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from code_context.config import get_settings
            log_path = get_settings().search_log_path
        except Exception:
            pass

    if not log_path:
        print("Error: SEARCH_LOG_PATH not configured. Set it in .env or use --log-path.")
        sys.exit(1)

    entries = _load_entries(log_path, last=args.last, since=args.since)
    analyze(entries)


if __name__ == "__main__":
    main()
