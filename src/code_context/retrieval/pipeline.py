"""Retrieval pipeline: Vector search + Reranking + Deduplication + Formatting."""

import logging
import re
from dataclasses import dataclass

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
        top_k: int | None = None,
        filepath: str | None = None,
        chunk_type: str | None = None,
        language: str | None = None,
        project: str | None = None,
        file_type: str | None = None,
        directory: str | None = None,
    ) -> list[SearchResult]:
        """Execute the full retrieval pipeline.

        Args:
            query: Natural language search query
            top_k: Number of results to return (default from settings)
            filepath: Filter to specific file
            chunk_type: Filter by chunk type (function, class, method)
            language: Filter by language
            project: Filter to specific project by its root path
            file_type: Filter by file type - "code" for code only, None/other for all
            directory: Filter to files within this directory path prefix

        Returns:
            List of SearchResult sorted by relevance (most relevant first)
        """
        top_k = top_k or self.settings.rerank_top_k_output

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

        if not candidates:
            logger.info(f"No candidates found for query: {query[:50]}...")
            return []

        # Filter by similarity threshold (permissive, for recall)
        sim_threshold = self.settings.similarity_threshold
        candidates = [c for c in candidates if c.similarity >= sim_threshold]

        if not candidates:
            logger.info(f"No candidates above similarity threshold {sim_threshold}")
            return []

        logger.debug(f"Phase 1: {len(candidates)} candidates above similarity {sim_threshold}")

        # ===== PHASE 2: Rerank =====
        documents = [c.chunk_text for c in candidates]
        rerank_results = await self.voyage.rerank(query, documents, top_k=len(candidates))

        # Map rerank results back to candidates
        reranked: list[tuple[ChunkResult, float]] = [
            (candidates[idx], score) for idx, score in rerank_results
        ]

        # ===== PHASE 3: Adaptive threshold filtering =====
        # Use adaptive threshold: min(config_threshold, top_score - margin)
        # This ensures we always return results for difficult queries
        top_score = reranked[0][1] if reranked else 0.0
        margin = 0.25  # Allow results within 0.25 of top score
        floor_threshold = 0.35  # Absolute minimum quality floor

        adaptive_threshold = max(floor_threshold, min(self.settings.rerank_threshold, top_score - margin))
        filtered = [(c, s) for c, s in reranked if s >= adaptive_threshold]

        if not filtered:
            # Fallback: take top 3 results if nothing passes even the adaptive threshold
            filtered = reranked[:3]
            logger.debug(f"No results above adaptive threshold {adaptive_threshold:.2f}, using top {len(filtered)}")

        logger.debug(
            f"Phase 3: {len(filtered)} results (adaptive threshold: {adaptive_threshold:.2f}, "
            f"top_score: {top_score:.2f}, config: {self.settings.rerank_threshold})"
        )

        # Stage 3: Deduplication (remove near-duplicates)
        deduplicated = self._deduplicate(filtered)

        # Stage 4: Format results with token budget
        max_tokens = self.settings.result_max_tokens
        current_tokens = 0
        results: list[SearchResult] = []

        for chunk, score in deduplicated[:top_k]:
            # Estimate tokens for this chunk (text + formatting overhead ~50 tokens)
            chunk_tokens = len(_encoder.encode(chunk.chunk_text)) + 50

            if current_tokens + chunk_tokens > max_tokens:
                logger.debug(f"Token budget reached ({current_tokens}/{max_tokens}), stopping at {len(results)} results")
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
            current_tokens += chunk_tokens

        # Sort by relevance (most relevant first - primacy effect)
        results.sort(key=lambda r: r.relevance_score, reverse=True)

        logger.debug(f"Returning {len(results)} results with ~{current_tokens} tokens")
        return results

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

    async def search_file(
        self,
        filepath: str,
        query: str,
        top_k: int = 5,
        project: str | None = None,
    ) -> list[SearchResult]:
        """Search within a specific file.

        Useful for local refactoring and understanding file context.
        """
        return await self.search(
            query=query,
            top_k=top_k,
            filepath=filepath,
            project=project,
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
