"""Retrieval pipeline: Vector search + Reranking + Deduplication + Formatting."""

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import tiktoken

from code_context.config import get_settings
from code_context.db.pool import DatabasePool
from code_context.db.schema import ChunkResult
from code_context.embedding.voyage import VoyageClient

logger = logging.getLogger(__name__)

# Token encoder for budget enforcement
_encoder = tiktoken.get_encoding("cl100k_base")

# Regex patterns for extracting module names from import statements
_IMPORT_PATTERNS = [
    re.compile(r'^import\s+(\w+)'),  # import foo
    re.compile(r'^from\s+([\w.]+)\s+import'),  # from foo.bar import
    re.compile(r"^import\s+['\"](.+?)['\"]"),  # import "foo" (JS/TS)
    re.compile(r"^from\s+['\"](.+?)['\"]"),  # from "foo" (JS/TS)
]

# Public search intents supported by MCP tools.
SearchIntent = Literal[
    "implementation",
    "definition",
    "usage",
    "debug",
    "security",
    "performance",
    "architecture",
]

_TEST_QUERY_PATTERN = re.compile(
    r"\b(test|tests|spec|e2e|unit\s+test|integration\s+test|fixture|mock)\b",
    re.IGNORECASE,
)

_TEST_FILE_PATH_PATTERN = re.compile(
    r"(^|/)(__tests__|tests?)/|\.test\.[^/]+$|\.spec\.[^/]+$",
    re.IGNORECASE,
)


def _extract_module_names(imports: list[str]) -> list[str]:
    """Extract just module names from import statements.

    Examples:
        'import { foo } from "drizzle-orm"' -> 'drizzle-orm'
        'from code_context.config import x' -> 'code_context'
        'import os' -> 'os'
    """
    modules: list[str] = []
    seen: set[str] = set()

    for imp in imports:
        imp = imp.strip()
        module = None

        for pattern in _IMPORT_PATTERNS:
            match = pattern.search(imp)
            if match:
                module = match.group(1)
                # Get just the top-level module
                module = module.split('.')[0]
                break

        if module and module not in seen:
            seen.add(module)
            modules.append(module)

    return modules


@dataclass
class SearchResult:
    """Final search result after retrieval pipeline."""

    filepath: str
    chunk_text: str
    chunk_type: str
    symbol_name: str | None
    start_line: int
    end_line: int
    relevance_score: float
    context_metadata: dict

    def format_for_context(self) -> str:
        """Format chunk for LLM context window.

        Follows the Contextual Retrieval Pattern from the SPEC:
        - File path and lines
        - Chunk type and symbol
        - Dependencies (imports)
        - The actual code
        """
        header_parts = [f"File: {self.filepath}:{self.start_line}-{self.end_line}"]

        if self.symbol_name:
            header_parts.append(f"Symbol: {self.symbol_name}")

        header_parts.append(f"Type: {self.chunk_type}")

        # Add imports if available (extract module names only)
        imports = self.context_metadata.get("imports", [])
        if imports:
            # Extract just module names from import statements
            modules = _extract_module_names(imports[:5])
            if modules:
                header_parts.append(f"Imports: {', '.join(modules)}")

        header = " | ".join(header_parts)

        return f"""```
{header}
---
{self.chunk_text}
```"""


class RetrievalPipeline:
    """Complete retrieval pipeline with vector search, reranking, and formatting.

    Pipeline stages:
    1. Vector Search: Query embedding + cosine similarity (top-k candidates)
    2. Reranking: rerank-2.5 for precision (filter by threshold)
    3. Deduplication: Remove semantically similar chunks
    4. Formatting: Prepare context for LLM consumption
    """

    def __init__(self, db: DatabasePool, voyage: VoyageClient):
        self.db = db
        self.voyage = voyage
        self.settings = get_settings()

    async def search(
        self,
        query: str,
        filepath: str | None = None,
        chunk_type: str | None = None,
        language: str | None = None,
        project: str | None = None,
        file_type: str | None = None,
        directory: str | None = None,
        search_intent: SearchIntent | None = None,
        max_tokens: int | None = None,
        include_tests: bool = False,
        max_file_chunks: int | None = 2,
        _tool_name: str | None = None,
    ) -> list[SearchResult]:
        """Execute the full retrieval pipeline.

        Args:
            query: Natural language search query
            filepath: Filter to specific file
            chunk_type: Filter by chunk type (function, class, method)
            language: Filter by language
            project: Filter to specific project by its root path
            file_type: Filter by file type - "code" for code only, None/other for all
            directory: Filter to files within this directory path prefix
            search_intent: Guides reranking toward a specific result type.
                "implementation" - concrete code that builds the feature
                "definition" - types, interfaces, schemas
                "usage" - call sites and integration examples
                "debug" - error handling, logging, edge cases
                "security" - auth, permissions, validation, secret handling
                "performance" - hotspots, caching, batching, throughput
                "architecture" - boundaries, orchestration, module contracts
            max_tokens: Optional per-request token budget (clamped to safe bounds)
            include_tests: Include test/spec files in final output (default: False)
            max_file_chunks: Cap number of generic file-level chunks (default: 2)
            _tool_name: Internal - tool name for quality logging

        Returns:
            List of SearchResult sorted by relevance (most relevant first)
        """
        t0 = time.monotonic()
        max_results = self.settings.rerank_top_k_output
        configured_max_tokens = self.settings.result_max_tokens
        requested_max_tokens = max_tokens
        effective_max_tokens = self._resolve_max_tokens(
            requested_max_tokens,
            configured_max_tokens,
        )
        effective_max_file_chunks = self._resolve_max_file_chunks(max_file_chunks)
        sim_threshold = self.settings.similarity_threshold
        phase0_count = 0
        phase1_count = 0
        resolved_intent = self._resolve_search_intent(search_intent)
        query_mentions_tests = self._query_explicitly_mentions_tests(query)
        file_chunks_selected = 0
        file_chunks_dropped = 0
        test_chunks_dropped = 0
        test_filter_applied = not include_tests
        test_filter_relaxed = False

        # Convert file_type to languages filter
        languages: list[str] | None = None
        if file_type == "code":
            languages = list(self.settings.code_languages)
        elif file_type == "docs":
            languages = list(self.settings.docs_languages)

        # Resolve relative directory to absolute path using project root
        abs_directory = None
        if directory and project:
            project_root = await self.db.get_project_root(project)
            if project_root:
                abs_directory = f"{project_root.rstrip('/')}/{directory.strip('/')}"

        # ===== PHASE 1: Vector Search (cast a wide net) =====
        query_embedding = await self.voyage.embed_query(query)

        candidates = await self.db.search_chunks(
            query_embedding=query_embedding,
            filepath=filepath,
            chunk_type=chunk_type,
            language=language,
            limit=self.settings.retrieval_k,  # 50 candidates
            project=project,
            languages=languages,
            directory=abs_directory,
        )
        phase0_count = len(candidates)

        if not candidates:
            duration_ms = int((time.monotonic() - t0) * 1000)
            logger.info(f"No candidates found for query: {query[:50]}...")
            self._log_search_quality(
                tool=_tool_name,
                query=query,
                project=project,
                filepath=filepath,
                chunk_type=chunk_type,
                language=language,
                file_type=file_type,
                search_intent=resolved_intent,
                directory=directory,
                query_mentions_tests=query_mentions_tests,
                outcome="no_candidates",
                reason="vector_search_returned_empty",
                phase0_candidates=phase0_count,
                phase1_candidates=phase1_count,
                sim_threshold=sim_threshold,
                top_score=0.0,
                threshold=0.0,
                results=[],
                result_token_counts=[],
                cut=[],
                total_tokens=0,
                requested_max_tokens=requested_max_tokens,
                effective_max_tokens=effective_max_tokens,
                max_results=max_results,
                deduplicated_count=0,
                include_tests=include_tests,
                max_file_chunks=effective_max_file_chunks,
                file_chunks_selected=file_chunks_selected,
                file_chunks_dropped=file_chunks_dropped,
                test_chunks_dropped=test_chunks_dropped,
                test_filter_applied=test_filter_applied,
                test_filter_relaxed=test_filter_relaxed,
                fallback_used=False,
                token_budget_exhausted=False,
                duration_ms=duration_ms,
            )
            return []

        # Filter by similarity threshold (permissive, for recall)
        candidates = [c for c in candidates if c.similarity >= sim_threshold]
        phase1_count = len(candidates)

        if not candidates:
            duration_ms = int((time.monotonic() - t0) * 1000)
            logger.info(f"No candidates above similarity threshold {sim_threshold}")
            self._log_search_quality(
                tool=_tool_name,
                query=query,
                project=project,
                filepath=filepath,
                chunk_type=chunk_type,
                language=language,
                file_type=file_type,
                search_intent=resolved_intent,
                directory=directory,
                query_mentions_tests=query_mentions_tests,
                outcome="below_similarity_threshold",
                reason=f"all_candidates_below_{sim_threshold:.2f}",
                phase0_candidates=phase0_count,
                phase1_candidates=phase1_count,
                sim_threshold=sim_threshold,
                top_score=0.0,
                threshold=0.0,
                results=[],
                result_token_counts=[],
                cut=[],
                total_tokens=0,
                requested_max_tokens=requested_max_tokens,
                effective_max_tokens=effective_max_tokens,
                max_results=max_results,
                deduplicated_count=0,
                include_tests=include_tests,
                max_file_chunks=effective_max_file_chunks,
                file_chunks_selected=file_chunks_selected,
                file_chunks_dropped=file_chunks_dropped,
                test_chunks_dropped=test_chunks_dropped,
                test_filter_applied=test_filter_applied,
                test_filter_relaxed=test_filter_relaxed,
                fallback_used=False,
                token_budget_exhausted=False,
                duration_ms=duration_ms,
            )
            return []

        logger.debug(f"Phase 1: {phase1_count} candidates above similarity {sim_threshold}")

        # ===== PHASE 2: Rerank with instruction-following =====
        documents = [c.chunk_text for c in candidates]
        rerank_query = self._build_rerank_query(query, resolved_intent)
        rerank_results = await self.voyage.rerank(rerank_query, documents, top_k=len(candidates))

        # Map rerank results back to candidates
        reranked: list[tuple[ChunkResult, float]] = [
            (candidates[idx], score) for idx, score in rerank_results
        ]

        # ===== PHASE 3: Relative threshold filtering =====
        top_score = reranked[0][1] if reranked else 0.0
        threshold = max(
            self.settings.rerank_score_floor,
            top_score * self.settings.rerank_relative_factor,
        )

        filtered = [(c, s) for c, s in reranked if s >= threshold]
        cut = [(c, s) for c, s in reranked if s < threshold]
        fallback_used = False

        if not filtered:
            # Fallback: take top 3 results if nothing passes threshold
            filtered = reranked[:3]
            cut = reranked[3:]
            fallback_used = True
            logger.debug(f"No results above threshold {threshold:.2f}, using top {len(filtered)}")

        logger.debug(
            f"Phase 3: {len(filtered)} results (threshold: {threshold:.2f}, "
            f"top_score: {top_score:.2f}, factor: {self.settings.rerank_relative_factor})"
        )

        # Stage 3: Deduplication (remove near-duplicates)
        deduplicated = self._deduplicate(filtered)
        deduplicated_count = len(deduplicated)

        # Stage 4: Optional test filtering + file chunk cap (token quality controls)
        ranked_candidates = deduplicated
        if test_filter_applied:
            non_test_candidates = [
                (chunk, score)
                for chunk, score in ranked_candidates
                if not self._is_test_file(chunk.filepath)
            ]
            test_chunks_dropped = len(ranked_candidates) - len(non_test_candidates)
            ranked_candidates = non_test_candidates

        ranked_candidates, file_chunks_selected, file_chunks_dropped = self._apply_file_chunk_cap(
            ranked_candidates,
            effective_max_file_chunks,
        )

        # Safety net: if filtering got too aggressive, relax test exclusion but keep file cap.
        if not ranked_candidates and test_filter_applied and deduplicated:
            test_filter_relaxed = True
            ranked_candidates, file_chunks_selected, file_chunks_dropped = self._apply_file_chunk_cap(
                deduplicated,
                effective_max_file_chunks,
            )
            test_chunks_dropped = 0

        # Stage 5: Format results with token budget
        current_tokens = 0
        results: list[SearchResult] = []
        result_token_counts: list[int] = []
        token_budget_exhausted = False

        for chunk, score in ranked_candidates:
            if len(results) >= max_results:
                break

            # Estimate tokens for this chunk (text + formatting overhead ~50 tokens)
            text_tokens = len(_encoder.encode(chunk.chunk_text))
            chunk_tokens = text_tokens + 50

            if current_tokens + chunk_tokens > effective_max_tokens:
                token_budget_exhausted = True
                logger.debug(
                    f"Token budget reached ({current_tokens}/{effective_max_tokens}), "
                    f"stopping at {len(results)} results"
                )
                break

            results.append(SearchResult(
                filepath=chunk.filepath,
                chunk_text=chunk.chunk_text,
                chunk_type=chunk.chunk_type,
                symbol_name=chunk.symbol_name,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                relevance_score=score,
                context_metadata=chunk.context_metadata,
            ))
            result_token_counts.append(text_tokens)
            current_tokens += chunk_tokens

        # Sort by relevance (most relevant first - primacy effect)
        results.sort(key=lambda r: r.relevance_score, reverse=True)

        duration_ms = int((time.monotonic() - t0) * 1000)
        logger.debug(f"Returning {len(results)} results with ~{current_tokens} tokens in {duration_ms}ms")

        outcome = "ok" if results else "empty_after_budget"
        reason = None
        if outcome == "empty_after_budget":
            if token_budget_exhausted:
                reason = "first_chunk_exceeded_budget"
            elif deduplicated_count and not ranked_candidates:
                reason = "no_results_after_controls"
            else:
                reason = "no_results_after_dedup"

        # Quality logging
        self._log_search_quality(
            tool=_tool_name,
            query=query,
            project=project,
            filepath=filepath,
            chunk_type=chunk_type,
            language=language,
            file_type=file_type,
            search_intent=resolved_intent,
            directory=directory,
            query_mentions_tests=query_mentions_tests,
            outcome=outcome,
            reason=reason,
            phase0_candidates=phase0_count,
            phase1_candidates=phase1_count,
            sim_threshold=sim_threshold,
            top_score=top_score,
            threshold=threshold,
            results=results,
            result_token_counts=result_token_counts,
            cut=cut,
            total_tokens=current_tokens,
            requested_max_tokens=requested_max_tokens,
            effective_max_tokens=effective_max_tokens,
            max_results=max_results,
            deduplicated_count=deduplicated_count,
            include_tests=include_tests,
            max_file_chunks=effective_max_file_chunks,
            file_chunks_selected=file_chunks_selected,
            file_chunks_dropped=file_chunks_dropped,
            test_chunks_dropped=test_chunks_dropped,
            test_filter_applied=test_filter_applied,
            test_filter_relaxed=test_filter_relaxed,
            fallback_used=fallback_used,
            token_budget_exhausted=token_budget_exhausted,
            duration_ms=duration_ms,
        )

        return results

    # Prompt version for quality-log traceability in A/B experiments.
    RERANK_PROMPT_VERSION = "v2.1"

    # Mapping from search_intent to rerank instruction (Voyage rerank-2.5 instruction-following).
    # Format follows Voyage recommendation: f"{instruction}\nRanking rules:\n...\nQuery: {query}"
    RERANK_INSTRUCTIONS: dict[str, str] = {
        "implementation": (
            "Prioritize code that directly implements the requested behavior and is likely to be edited "
            "to deliver the feature. Prefer concrete runtime logic (services, handlers, orchestrators, "
            "repositories with side effects) over broad references."
        ),
        "definition": (
            "Prioritize canonical declarations: interfaces, types, schemas, enums, contracts, and "
            "configuration structures that define data shape and constraints. Prefer definitions over usage."
        ),
        "usage": (
            "Prioritize call sites and integration points showing how the target is invoked, wired, "
            "or consumed across modules. Prefer consumers over providers."
        ),
        "debug": (
            "Prioritize failure-path code: error handling, retries, timeouts, fallbacks, guards, "
            "validation failures, and observability signals that explain root cause propagation."
        ),
        "security": (
            "Prioritize security-critical code: authentication, authorization, RBAC checks, token/session "
            "handling, secret management, input validation/sanitization, and injection protections."
        ),
        "performance": (
            "Prioritize performance-critical paths: hot loops, expensive I/O, query shape, batching, "
            "caching, queue throughput, contention points, and memory-heavy transformations."
        ),
        "architecture": (
            "Prioritize architectural boundaries and flow orchestration: module contracts, interfaces "
            "between subsystems, adapters, factories, and cross-layer integration paths."
        ),
    }

    INTENT_ALIASES: dict[str, str] = {
        "impl": "implementation",
        "types": "definition",
        "schema": "definition",
        "examples": "usage",
        "calls": "usage",
        "bugfix": "debug",
        "troubleshoot": "debug",
        "perf": "performance",
        "speed": "performance",
        "design": "architecture",
        "arch": "architecture",
        "authz": "security",
        "auth": "security",
    }

    COMMON_RANKING_RULES: tuple[str, ...] = (
        "Prefer chunks that directly answer the query over generic mentions.",
        "Prefer symbol-scoped chunks (function/method/class/declaration) over generic file-level chunks "
        "when both are relevant.",
        "Penalize low-signal boilerplate and unrelated utilities.",
    )

    # Default intent remains implementation for backward compatibility.
    DEFAULT_RERANK_INSTRUCTION = RERANK_INSTRUCTIONS["implementation"]

    @staticmethod
    def _resolve_search_intent(search_intent: str | None) -> SearchIntent:
        """Normalize/resolve search intent with safe defaulting."""
        if not search_intent:
            return "implementation"

        normalized = search_intent.strip().lower()
        normalized = RetrievalPipeline.INTENT_ALIASES.get(normalized, normalized)

        if normalized in RetrievalPipeline.RERANK_INSTRUCTIONS:
            return cast(SearchIntent, normalized)
        return "implementation"

    @staticmethod
    def _query_explicitly_mentions_tests(query: str) -> bool:
        """Heuristic: keep tests high only when query asks for tests explicitly."""
        return _TEST_QUERY_PATTERN.search(query) is not None

    @staticmethod
    def _is_test_file(filepath: str) -> bool:
        """Detect whether a file path is likely test/spec code."""
        normalized = filepath.replace("\\", "/")
        return _TEST_FILE_PATH_PATTERN.search(normalized) is not None

    @staticmethod
    def _resolve_max_tokens(
        requested_max_tokens: int | None,
        configured_max_tokens: int,
    ) -> int:
        """Resolve per-request token budget with safety clamps."""
        if requested_max_tokens is None:
            return configured_max_tokens

        # Respect configured ceiling even when it's lower than the default floor.
        if configured_max_tokens < 256:
            return max(1, min(requested_max_tokens, configured_max_tokens))

        return max(256, min(requested_max_tokens, configured_max_tokens))

    @staticmethod
    def _resolve_max_file_chunks(max_file_chunks: int | None) -> int | None:
        """Resolve per-request cap for generic file-level chunks."""
        if max_file_chunks is None:
            return None
        return max(0, max_file_chunks)

    @staticmethod
    def _apply_file_chunk_cap(
        ranked_candidates: list[tuple[ChunkResult, float]],
        max_file_chunks: int | None,
    ) -> tuple[list[tuple[ChunkResult, float]], int, int]:
        """Keep ranked order while limiting how many file-level chunks are returned."""
        if max_file_chunks is None:
            selected = sum(1 for chunk, _ in ranked_candidates if chunk.chunk_type == "file")
            return ranked_candidates, selected, 0

        kept: list[tuple[ChunkResult, float]] = []
        selected = 0
        dropped = 0
        for chunk, score in ranked_candidates:
            if chunk.chunk_type == "file":
                if selected >= max_file_chunks:
                    dropped += 1
                    continue
                selected += 1
            kept.append((chunk, score))

        return kept, selected, dropped

    @staticmethod
    def _build_rerank_query(
        query: str,
        search_intent: SearchIntent | str | None = None,
    ) -> str:
        """Prepend a reranking instruction to the query.

        rerank-2.5 supports instruction-following: natural language instructions
        prepended to the query steer relevance scoring. Uses the Voyage-recommended
        format: "{instruction}\\nRanking rules:\\n...\\nQuery: {query}".
        """
        resolved_intent = RetrievalPipeline._resolve_search_intent(search_intent)
        instruction = RetrievalPipeline.RERANK_INSTRUCTIONS[resolved_intent]

        ranking_rules = list(RetrievalPipeline.COMMON_RANKING_RULES)

        if not RetrievalPipeline._query_explicitly_mentions_tests(query):
            ranking_rules.append(
                "Deprioritize test/spec/fixture/mocked code unless no stronger production evidence exists."
            )

        if resolved_intent in {"definition", "architecture"}:
            ranking_rules.append(
                "Strongly prefer declarations and contracts over broad implementation bodies."
            )
        elif resolved_intent in {"usage", "debug"}:
            ranking_rules.append(
                "Prefer execution-path evidence (callers, handlers, orchestration paths) over static definitions."
            )

        rules_block = "\n".join(f"- {rule}" for rule in ranking_rules)
        return f"{instruction}\nRanking rules:\n{rules_block}\nQuery: {query}"

    def _deduplicate(
        self,
        results: list[tuple[ChunkResult, float]],
        text_similarity_threshold: float = 0.85,
        line_overlap_threshold: float = 0.25,
    ) -> list[tuple[ChunkResult, float]]:
        """Remove redundant chunks using line overlap and text similarity.

        Three-phase deduplication:
        1. Line overlap: If chunks from the same file have >25% line overlap,
           keep only the higher-scored one. This aggressively removes near-duplicates.
        2. Containment: If chunk A fully contains chunk B, keep the higher-scored one.
        3. Text similarity: If two chunks have >85% text overlap, keep the higher scored one.
        """
        if len(results) <= 1:
            return results

        deduplicated: list[tuple[ChunkResult, float]] = []
        # Track (filepath, start_line, end_line, score, index) for overlap checks
        seen_ranges: list[tuple[str, int, int, float, int]] = []
        seen_texts: set[str] = set()

        for chunk, score in results:
            filepath = chunk.filepath
            start = chunk.start_line
            end = chunk.end_line
            chunk_lines = end - start + 1

            # Phase 1 & 2: Line overlap and containment check
            should_skip = False
            replace_index = None

            for i, (seen_path, seen_start, seen_end, seen_score, dedup_idx) in enumerate(seen_ranges):
                if seen_path != filepath:
                    continue

                seen_lines = seen_end - seen_start + 1

                # Calculate overlap
                overlap_start = max(start, seen_start)
                overlap_end = min(end, seen_end)
                overlap_lines = max(0, overlap_end - overlap_start + 1)

                # Check overlap ratio for both chunks
                current_overlap_ratio = overlap_lines / chunk_lines if chunk_lines > 0 else 0
                seen_overlap_ratio = overlap_lines / seen_lines if seen_lines > 0 else 0
                max_overlap_ratio = max(current_overlap_ratio, seen_overlap_ratio)

                if max_overlap_ratio >= line_overlap_threshold:
                    # Significant overlap - keep only the higher-scored one
                    if score > seen_score:
                        replace_index = i
                        break
                    else:
                        should_skip = True
                        break

            if should_skip:
                continue

            # If current has higher score than an overlapping chunk, replace it
            if replace_index is not None:
                old_dedup_idx = seen_ranges[replace_index][4]
                old_chunk = deduplicated[old_dedup_idx][0]
                # Mark for removal (we'll clean up after)
                seen_ranges.pop(replace_index)
                deduplicated[old_dedup_idx] = None  # type: ignore
                # Remove old text from seen_texts
                old_normalized = old_chunk.chunk_text.strip().lower()
                seen_texts.discard(old_normalized)

            # Phase 3: Text similarity check (for chunks from different files)
            normalized = chunk.chunk_text.strip().lower()
            is_text_duplicate = False
            for seen in seen_texts:
                if self._text_similarity(normalized, seen) > text_similarity_threshold:
                    is_text_duplicate = True
                    break

            if not is_text_duplicate:
                new_idx = len(deduplicated)
                deduplicated.append((chunk, score))
                seen_ranges.append((filepath, start, end, score, new_idx))
                seen_texts.add(normalized)

        # Clean up None entries from replacements
        return [r for r in deduplicated if r is not None]

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple Jaccard similarity for text deduplication."""
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _log_search_quality(
        self,
        *,
        tool: str | None,
        query: str,
        project: str | None,
        filepath: str | None,
        chunk_type: str | None,
        language: str | None,
        file_type: str | None,
        search_intent: str | None,
        directory: str | None,
        query_mentions_tests: bool,
        outcome: str,
        reason: str | None,
        phase0_candidates: int,
        phase1_candidates: int,
        sim_threshold: float,
        top_score: float,
        threshold: float,
        results: list[SearchResult],
        result_token_counts: list[int],
        cut: list[tuple[ChunkResult, float]],
        total_tokens: int,
        requested_max_tokens: int | None,
        effective_max_tokens: int,
        max_results: int,
        deduplicated_count: int,
        include_tests: bool,
        max_file_chunks: int | None,
        file_chunks_selected: int,
        file_chunks_dropped: int,
        test_chunks_dropped: int,
        test_filter_applied: bool,
        test_filter_relaxed: bool,
        fallback_used: bool,
        token_budget_exhausted: bool,
        duration_ms: int,
    ) -> None:
        """Append a quality log entry to JSONL file (best-effort, never fails the search)."""
        log_path = self.settings.search_log_path
        if not log_path:
            return

        try:
            entry = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "tool": tool,
                "query": query,
                "outcome": outcome,
                "reason": reason,
                "filters": {
                    "project": project,
                    "filepath": filepath,
                    "chunk_type": chunk_type,
                    "language": language,
                    "file_type": file_type,
                    "search_intent": search_intent,
                    "directory": directory,
                    "include_tests": include_tests,
                    "query_mentions_tests": query_mentions_tests,
                },
                "retrieval": {
                    "rerank_prompt_version": self.RERANK_PROMPT_VERSION,
                    "phase0_candidates": phase0_candidates,
                    "phase1_candidates": phase1_candidates,
                    "similarity_threshold": round(sim_threshold, 4),
                    "top_score": round(top_score, 4),
                    "threshold": round(threshold, 4),
                    "deduplicated_count": deduplicated_count,
                    "test_filter_applied": test_filter_applied,
                    "test_filter_relaxed": test_filter_relaxed,
                    "test_chunks_dropped": test_chunks_dropped,
                    "max_file_chunks": max_file_chunks,
                    "file_chunks_selected": file_chunks_selected,
                    "file_chunks_dropped": file_chunks_dropped,
                    "fallback_used": fallback_used,
                    "cut_count": len(cut),
                    "cut_scores": [round(s, 4) for _, s in cut[:10]],
                },
                "budget": {
                    "max_results": max_results,
                    "requested_max_tokens": requested_max_tokens,
                    "max_tokens": effective_max_tokens,
                    "used_tokens": total_tokens,
                    "returned_results": len(results),
                    "token_budget_exhausted": token_budget_exhausted,
                },
                "results": [
                    {
                        "file": r.filepath,
                        "symbol": r.symbol_name,
                        "type": r.chunk_type,
                        "lines": f"{r.start_line}-{r.end_line}",
                        "score": round(r.relevance_score, 4),
                        "tokens": (
                            result_token_counts[i]
                            if i < len(result_token_counts)
                            else len(_encoder.encode(r.chunk_text))
                        ),
                    }
                    for i, r in enumerate(results)
                ],
                "duration_ms": duration_ms,
            }

            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            logger.debug("Failed to write search quality log", exc_info=True)

    async def search_file(
        self,
        filepath: str,
        query: str,
        project: str | None = None,
        search_intent: SearchIntent | None = None,
        max_tokens: int | None = None,
        include_tests: bool = False,
        max_file_chunks: int | None = 2,
    ) -> list[SearchResult]:
        """Search within a specific file.

        Useful for local refactoring and understanding file context.
        """
        return await self.search(
            query=query,
            filepath=filepath,
            project=project,
            search_intent=search_intent,
            max_tokens=max_tokens,
            include_tests=include_tests,
            max_file_chunks=max_file_chunks,
            _tool_name="search_by_file",
        )


def format_results_for_llm(results: list[SearchResult]) -> str:
    """Format search results for LLM context window.

    Positions most relevant chunks at the BEGINNING (primacy effect).
    """
    if not results:
        return "No relevant code found."

    formatted = []
    for i, result in enumerate(results, 1):
        formatted.append(f"## Result {i} (relevance: {result.relevance_score:.2f})\n")
        formatted.append(result.format_for_context())
        formatted.append("")

    return "\n".join(formatted)
