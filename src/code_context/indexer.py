"""Indexer - Handles file indexing and embedding generation."""

import asyncio
import concurrent.futures
import logging
import os
from datetime import datetime
from pathlib import Path

from code_context.chunking.parser import (
    CodeParser,
    ParsedChunk,
    compute_chunk_hash,
    compute_file_hash,
    detect_language,
)
from code_context.config import get_settings
from code_context.db.pool import DatabasePool
from code_context.db.schema import CodeChunk, CodeFile
from code_context.embedding.voyage import VoyageClient

logger = logging.getLogger(__name__)

# Default ignore patterns
DEFAULT_IGNORE_PATTERNS = {
    # Directories
    "node_modules",
    ".git",
    ".svn",
    ".hg",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "venv",
    ".venv",
    "env",
    ".env",
    "dist",
    "build",
    "target",
    ".next",
    ".nuxt",
    ".trigger",
    ".turbo",
    ".vercel",
    ".wrangler",
    ".alchemy",
    ".sst",
    ".open-next",
    ".angular",
    ".svelte-kit",
    ".astro",
    ".parcel-cache",
    ".cache",
    ".output",
    ".nitro",
    ".vinxi",
    "storybook-static",
    ".docker",
    ".terraform",
    "coverage",
    ".nyc_output",
    # Report/generated directories
    "reports",
    ".report",
    ".reports",
    "test-results",
    "playwright-report",
    # Files
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Cargo.lock",
    ".DS_Store",
}

# File patterns to always ignore (checked against filename)
IGNORE_FILE_PATTERNS = {
    # Reports and generated files (often huge)
    "-report-",
    ".report.",
    ".min.js",
    ".min.css",
    ".bundle.js",
    ".chunk.js",
    # Source maps
    ".map",
}


class Indexer:
    """Handles incremental indexing of codebases."""

    def __init__(
        self,
        db: DatabasePool,
        voyage: VoyageClient | None,
        ignore_patterns: set[str] | None = None,
    ):
        self.db = db
        self.voyage = voyage
        self.parser = CodeParser()
        self.settings = get_settings()
        self.ignore_patterns = ignore_patterns or DEFAULT_IGNORE_PATTERNS

    def should_ignore(self, path: Path) -> bool:
        """Check if path should be ignored."""
        parts = path.parts
        filename = path.name

        # Check directory patterns
        for pattern in self.ignore_patterns:
            if pattern in parts or filename == pattern:
                return True

        # Check file patterns (substring match in filename)
        for pattern in IGNORE_FILE_PATTERNS:
            if pattern in filename:
                return True

        return False

    @staticmethod
    def _build_embedding_text(chunk: ParsedChunk, language: str) -> str:
        """Build a contextualized text representation for embedding.

        Keeps original chunk text intact in storage/output, but prepends compact
        structural context to improve semantic + identifier retrieval quality.
        """
        context = chunk.context or {}
        filepath = context.get("filepath", "")
        imports = context.get("imports", []) or []
        parent_class = context.get("parent_class")

        header_parts = [
            f"filepath={filepath}",
            f"language={language}",
            f"chunk_type={chunk.chunk_type}",
        ]

        if chunk.symbol_name:
            header_parts.append(f"symbol={chunk.symbol_name}")
        if parent_class:
            header_parts.append(f"parent_class={parent_class}")
        if imports:
            header_parts.append(f"imports={', '.join(imports[:5])}")

        header = " | ".join(header_parts)
        return f"[context] {header}\n{chunk.text}"

    def _collect_files(self, root: Path) -> list[tuple[Path, str]]:
        """Collect indexable files using os.walk with directory pruning.

        ~450x faster than rglob for projects with large ignored dirs (node_modules).
        """
        files: list[tuple[Path, str]] = []
        for dirpath, dirnames, filenames in os.walk(root):
            # Prune ignored directories in-place (prevents descending)
            dirnames[:] = [d for d in dirnames if d not in self.ignore_patterns]

            for filename in filenames:
                # Check file-level ignore patterns
                if any(p in filename for p in IGNORE_FILE_PATTERNS):
                    continue
                if filename in self.ignore_patterns:
                    continue

                filepath = os.path.join(dirpath, filename)
                language = detect_language(filepath)
                if language:
                    files.append((Path(filepath), language))

        return files

    async def index_project(
        self,
        project_root: str,
        project_id: str,
        force: bool = False,
        ensure_vector_index: bool = True,
    ) -> dict:
        """Index all supported files in a project.

        Args:
            project_root: Root directory of the project
            project_id: Short identifier for the project (e.g., "my-app")
            force: If True, reindex all files regardless of hash
            ensure_vector_index: If True, ensure vector index exists after changes

        Returns:
            Statistics about the indexing operation
        """
        root = Path(project_root).resolve()
        if not root.is_dir():
            raise ValueError(f"Not a directory: {project_root}")

        stats = {
            "project_id": project_id,
            "total_files": 0,
            "indexed_files": 0,
            "skipped_files": 0,
            "deleted_files": 0,
            "total_chunks": 0,
            "errors": [],
        }

        # Collect eligible files on disk (os.walk with pruning, ~450x faster than rglob)
        files_to_index = self._collect_files(root)
        disk_filepaths = {str(p) for p, _ in files_to_index}
        stats["total_files"] = len(files_to_index)

        # Remove stale files from DB (deleted from disk or now ignored)
        indexed_files = await self.db.get_project_files(project_id)
        stale_files = set(indexed_files.keys()) - disk_filepaths
        if stale_files:
            try:
                deleted_count = await self.db.delete_files(sorted(stale_files))
                stats["deleted_files"] = deleted_count
                logger.info(f"Removed {deleted_count} stale files")
            except Exception as e:
                logger.error(f"Error removing stale files: {e}")

        logger.info(
            f"Found {stats['total_files']} files to index in {project_root} (id: {project_id})"
            + (f", removed {stats['deleted_files']} stale" if stats["deleted_files"] else "")
        )

        # Process files concurrently with semaphore
        semaphore = asyncio.Semaphore(self.settings.index_concurrency)

        async def _process_file(path: Path, language: str) -> dict:
            async with semaphore:
                try:
                    return await self.index_file(
                        filepath=str(path),
                        project_root=str(root),
                        project_id=project_id,
                        language=language,
                        force=force,
                        known_file_hash=indexed_files.get(str(path)),
                    )
                except Exception as e:
                    logger.error(f"Error indexing {path}: {e}")
                    return {"error": str(e), "file": str(path)}

        tasks = [_process_file(path, lang) for path, lang in files_to_index]
        results = await asyncio.gather(*tasks)

        for result, (path, _) in zip(results, files_to_index):
            if "error" in result:
                stats["errors"].append({"file": str(path), "error": result["error"]})
            elif result["indexed"]:
                stats["indexed_files"] += 1
                stats["total_chunks"] += result["chunks"]
            else:
                stats["skipped_files"] += 1

        # Create vector index if we indexed anything
        if ensure_vector_index and (stats["indexed_files"] > 0 or stats["deleted_files"] > 0):
            try:
                await self.db.create_vector_index()
            except Exception as e:
                logger.warning(f"Could not create vector index: {e}")

        return stats

    async def index_file(
        self,
        filepath: str,
        project_root: str,
        project_id: str,
        language: str | None = None,
        force: bool = False,
        known_file_hash: str | None = None,
    ) -> dict:
        """Index a single file.

        Args:
            filepath: Path to the file
            project_root: Root directory of the project
            project_id: Short identifier for the project
            language: Language of the file (auto-detected if not provided)
            force: If True, reindex regardless of hash
            known_file_hash: Optional preloaded hash from project scan to avoid per-file DB lookups

        Returns:
            Dict with indexing results
        """
        path = Path(filepath)
        if not path.is_file():
            raise ValueError(f"Not a file: {filepath}")

        language = language or detect_language(filepath)
        if not language:
            return {"indexed": False, "reason": "unsupported_language", "chunks": 0}

        # Check file size
        file_size_kb = path.stat().st_size / 1024
        max_size = self.settings.max_file_size_kb
        if file_size_kb > max_size:
            logger.warning(f"Skipping large file ({file_size_kb:.0f}KB > {max_size}KB): {filepath}")
            return {"indexed": False, "reason": "file_too_large", "chunks": 0}

        # Read file content
        content = path.read_text(encoding="utf-8")
        file_hash = compute_file_hash(content)

        # Check if file needs reindexing
        existing_hash = known_file_hash
        if not force and existing_hash is None:
            existing = await self.db.get_file(filepath)
            existing_hash = existing.file_hash if existing else None

        if not force and existing_hash == file_hash:
            return {"indexed": False, "reason": "unchanged", "chunks": 0}

        # Parse file into chunks
        parsed_chunks = self.parser.parse_file(filepath, content)

        # Filter out chunks with empty text (Voyage API rejects them)
        parsed_chunks = [c for c in parsed_chunks if c.text and c.text.strip()]

        if not parsed_chunks:
            # Clean up any stale DB entry (safe even if file not in DB)
            await self.db.delete_file(filepath)
            return {"indexed": False, "reason": "no_chunks", "chunks": 0}

        # Generate embeddings for all chunks
        texts = [self._build_embedding_text(chunk, language) for chunk in parsed_chunks]
        embeddings = await self.voyage.embed_documents(texts, input_type="document")

        # Create chunk objects
        chunks = [
            CodeChunk(
                filepath=filepath,
                chunk_hash=compute_chunk_hash(chunk.text, filepath, chunk.start_line),
                chunk_text=chunk.text,
                embedding=embedding,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                chunk_type=chunk.chunk_type,
                symbol_name=chunk.symbol_name,
                context_metadata=chunk.context,
                token_count=chunk.token_count,
            )
            for chunk, embedding in zip(parsed_chunks, embeddings)
        ]

        # Atomic: delete old chunks + upsert file + insert new chunks
        # Single transaction — Ctrl+C rolls back cleanly
        file_record = CodeFile(
            filepath=filepath,
            file_hash=file_hash,
            language=language,
            project_root=project_root,
            project_id=project_id,
            last_indexed=datetime.now(),
            loc=content.count("\n") + 1,
        )
        await self.db.reindex_file_atomic(file_record, chunks)

        logger.info(f"Indexed {filepath}: {len(chunks)} chunks")

        return {"indexed": True, "chunks": len(chunks)}

    def _is_indexable(self, filepath: str) -> bool:
        """Check if a file would actually be indexed (size limit + produces chunks).

        Replicates the same eligibility checks as index_file without calling
        the Voyage API. Fast: only does local I/O and tree-sitter parsing.
        """
        path = Path(filepath)
        try:
            file_size_kb = path.stat().st_size / 1024
            if file_size_kb > self.settings.max_file_size_kb:
                return False
            content = path.read_text(encoding="utf-8")
            chunks = self.parser.parse_file(filepath, content)
            return any(c.text and c.text.strip() for c in chunks)
        except Exception:
            return False

    async def check_status(
        self,
        project_root: str,
        project_id: str,
    ) -> dict:
        """Check sync status without modifying anything (dry-run).

        Compares files on disk with the indexed state in the database.
        Applies the same eligibility filters as index_file.

        Optimized: only parses genuinely new files, uses thread pool for
        parallel hash I/O on existing files.
        """
        root = Path(project_root).resolve()
        if not root.is_dir():
            raise ValueError(f"Not a directory: {project_root}")

        # 1. DB query + disk scan (os.walk with pruning, no file content reads)
        indexed_files = await self.db.get_project_files(project_id)
        indexed_set = set(indexed_files.keys())

        collected = self._collect_files(root)
        candidate_set = {str(p) for p, _ in collected}

        # 2. Split into categories
        potentially_new = candidate_set - indexed_set  # need parse check
        deleted_files: list[str] = sorted(indexed_set - candidate_set)
        common = candidate_set & indexed_set  # need hash check only

        # 3. New files: check if indexable (parse) — only these need it
        skipped_ineligible = 0
        new_files: list[str] = []
        for filepath in sorted(potentially_new):
            if self._is_indexable(filepath):
                new_files.append(filepath)
            else:
                skipped_ineligible += 1

        # 4. Common files: parallel hash comparison (I/O-bound)
        modified_files: list[str] = []
        unchanged_files: list[str] = []

        def _check_hash(filepath: str) -> tuple[str, bool]:
            """Read file and compare hash. Returns (filepath, changed)."""
            try:
                content = Path(filepath).read_text(encoding="utf-8")
                return (filepath, compute_file_hash(content) != indexed_files[filepath])
            except Exception:
                return (filepath, True)  # unreadable = treat as changed

        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            results = await asyncio.gather(
                *[loop.run_in_executor(pool, _check_hash, fp) for fp in common]
            )

        for filepath, changed in results:
            if changed:
                modified_files.append(filepath)
            else:
                unchanged_files.append(filepath)

        modified_files.sort()
        unchanged_files.sort()

        eligible_on_disk = len(common) + len(new_files)
        up_to_date = not new_files and not modified_files and not deleted_files

        return {
            "project_id": project_id,
            "project_root": project_root,
            "up_to_date": up_to_date,
            "new": new_files,
            "modified": modified_files,
            "deleted": deleted_files,
            "unchanged": unchanged_files,
            "skipped_ineligible": skipped_ineligible,
            "total_on_disk": eligible_on_disk,
            "total_indexed": len(indexed_files),
        }

    async def remove_file(self, filepath: str) -> bool:
        """Remove a file from the index."""
        await self.db.delete_file(filepath)
        logger.info(f"Removed {filepath} from index")
        return True

    async def get_stats(self) -> dict:
        """Get indexing statistics."""
        return await self.db.get_index_stats()
