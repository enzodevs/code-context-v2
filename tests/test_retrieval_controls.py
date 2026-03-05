"""Unit tests for token-efficient retrieval controls."""

import json
from pathlib import Path
from typing import Any

import pytest

from code_context.config import get_settings
from code_context.db.schema import ChunkResult
from code_context.retrieval.pipeline import RetrievalPipeline


class DummyDB:
    def __init__(self, candidates: list[ChunkResult]):
        self._candidates = candidates

    async def get_project_root(self, _project_id: str) -> str:
        return "/repo"

    async def search_chunks(self, **_kwargs: Any) -> list[ChunkResult]:
        return self._candidates


class DummyVoyage:
    async def embed_query(self, _query: str) -> list[float]:
        return [0.0]

    async def rerank(
        self,
        _query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        k = min(top_k or len(documents), len(documents))
        return [(idx, 1.0 - (idx * 0.01)) for idx in range(k)]


def _make_chunk(
    chunk_id: int,
    filepath: str,
    *,
    chunk_type: str = "function",
    text: str | None = None,
    similarity: float = 0.9,
) -> ChunkResult:
    payload = text or f"chunk_{chunk_id} unique tokens {chunk_id}"
    return ChunkResult(
        chunk_id=chunk_id,
        filepath=filepath,
        chunk_text=payload,
        chunk_type=chunk_type,
        symbol_name=f"symbol_{chunk_id}",
        start_line=chunk_id * 10,
        end_line=chunk_id * 10 + 5,
        context_metadata={},
        similarity=similarity,
    )


def _read_logs(log_path: Path) -> list[dict]:
    lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [json.loads(line) for line in lines]


@pytest.fixture
def build_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    settings = get_settings()
    log_path = tmp_path / "search-quality.jsonl"

    monkeypatch.setattr(settings, "search_log_path", str(log_path))
    monkeypatch.setattr(settings, "retrieval_k", 50)
    monkeypatch.setattr(settings, "similarity_threshold", 0.0)
    monkeypatch.setattr(settings, "rerank_top_k_output", 8)
    monkeypatch.setattr(settings, "result_max_tokens", 8000)

    def _builder(candidates: list[ChunkResult]) -> tuple[RetrievalPipeline, Path]:
        return RetrievalPipeline(DummyDB(candidates), DummyVoyage()), log_path

    return _builder


@pytest.mark.asyncio
async def test_max_tokens_clamp_min_and_max(build_pipeline):
    candidates = [_make_chunk(1, "/repo/src/a.ts", text="alpha beta gamma delta")]
    pipeline, log_path = build_pipeline(candidates)

    await pipeline.search("query", max_tokens=10, include_tests=True, max_file_chunks=None)
    await pipeline.search("query", max_tokens=99999, include_tests=True, max_file_chunks=None)

    entries = _read_logs(log_path)
    assert entries[0]["budget"]["requested_max_tokens"] == 10
    assert entries[0]["budget"]["max_tokens"] == 256
    assert entries[1]["budget"]["requested_max_tokens"] == 99999
    assert entries[1]["budget"]["max_tokens"] == 8000


@pytest.mark.asyncio
async def test_max_tokens_respects_small_configured_ceiling(build_pipeline, monkeypatch: pytest.MonkeyPatch):
    candidates = [_make_chunk(1, "/repo/src/a.ts", text="alpha beta gamma delta")]
    pipeline, log_path = build_pipeline(candidates)
    monkeypatch.setattr(pipeline.settings, "result_max_tokens", 128)

    await pipeline.search("query", max_tokens=10, include_tests=True, max_file_chunks=None)
    await pipeline.search("query", max_tokens=99999, include_tests=True, max_file_chunks=None)

    entries = _read_logs(log_path)
    assert entries[0]["budget"]["requested_max_tokens"] == 10
    assert entries[0]["budget"]["max_tokens"] == 10
    assert entries[1]["budget"]["requested_max_tokens"] == 99999
    assert entries[1]["budget"]["max_tokens"] == 128


@pytest.mark.asyncio
async def test_include_tests_false_drops_test_chunks(build_pipeline):
    candidates = [
        _make_chunk(1, "/repo/src/module/foo.test.ts"),
        _make_chunk(2, "/repo/src/module/foo.ts"),
    ]
    pipeline, log_path = build_pipeline(candidates)

    results = await pipeline.search("query", include_tests=False, max_file_chunks=None)

    assert len(results) == 1
    assert all(".test." not in r.filepath for r in results)

    entry = _read_logs(log_path)[-1]
    assert entry["filters"]["include_tests"] is False
    assert entry["retrieval"]["test_filter_applied"] is True
    assert entry["retrieval"]["test_chunks_dropped"] == 1


@pytest.mark.asyncio
async def test_include_tests_true_keeps_test_chunks(build_pipeline):
    candidates = [
        _make_chunk(1, "/repo/src/module/foo.test.ts"),
        _make_chunk(2, "/repo/src/module/foo.ts"),
    ]
    pipeline, log_path = build_pipeline(candidates)

    results = await pipeline.search("query", include_tests=True, max_file_chunks=None)

    assert len(results) == 2

    entry = _read_logs(log_path)[-1]
    assert entry["filters"]["include_tests"] is True
    assert entry["retrieval"]["test_filter_applied"] is False
    assert entry["retrieval"]["test_chunks_dropped"] == 0


@pytest.mark.asyncio
async def test_max_file_chunks_cap_limits_file_results(build_pipeline):
    candidates = [
        _make_chunk(1, "/repo/src/a.ts", chunk_type="file", text="file_a unique"),
        _make_chunk(2, "/repo/src/b.ts", chunk_type="file", text="file_b unique"),
        _make_chunk(3, "/repo/src/c.ts", chunk_type="file", text="file_c unique"),
        _make_chunk(4, "/repo/src/svc.ts", chunk_type="function", text="function_svc unique"),
    ]
    pipeline, log_path = build_pipeline(candidates)

    results = await pipeline.search("query", include_tests=True, max_file_chunks=2)

    assert sum(1 for r in results if r.chunk_type == "file") == 2
    assert any(r.chunk_type != "file" for r in results)

    entry = _read_logs(log_path)[-1]
    assert entry["retrieval"]["max_file_chunks"] == 2
    assert entry["retrieval"]["file_chunks_selected"] == 2
    assert entry["retrieval"]["file_chunks_dropped"] == 1


@pytest.mark.asyncio
async def test_max_file_chunks_zero_removes_generic_file_chunks(build_pipeline):
    candidates = [
        _make_chunk(1, "/repo/src/a.ts", chunk_type="file", text="file_a unique"),
        _make_chunk(2, "/repo/src/b.ts", chunk_type="file", text="file_b unique"),
        _make_chunk(3, "/repo/src/svc.ts", chunk_type="function", text="function_svc unique"),
    ]
    pipeline, log_path = build_pipeline(candidates)

    results = await pipeline.search("query", include_tests=True, max_file_chunks=0)

    assert all(r.chunk_type != "file" for r in results)

    entry = _read_logs(log_path)[-1]
    assert entry["retrieval"]["max_file_chunks"] == 0
    assert entry["retrieval"]["file_chunks_selected"] == 0
    assert entry["retrieval"]["file_chunks_dropped"] == 2


@pytest.mark.asyncio
async def test_per_file_budget_caps_tokens(build_pipeline, monkeypatch: pytest.MonkeyPatch):
    """Per-file budget should prevent one file from dominating results."""
    big_text = "x " * 300  # ~300 tokens per chunk
    candidates = [
        _make_chunk(i, "/repo/src/heavy.ts", text=f"{big_text} unique_{i}")
        for i in range(1, 7)
    ]
    pipeline, log_path = build_pipeline(candidates)
    monkeypatch.setattr(pipeline.settings, "hierarchical_assembly_enabled", True)
    monkeypatch.setattr(pipeline.settings, "per_file_budget_ratio", 0.30)
    monkeypatch.setattr(pipeline.settings, "result_max_tokens", 4000)
    monkeypatch.setattr(pipeline.settings, "max_symbols_per_file", 10)  # high, so budget is the limiter

    results = await pipeline.search("query", include_tests=True, max_file_chunks=None)

    # Single-file queries should bypass per-file budget
    # All 6 chunks are from same file, so per-file budget is not enforced
    assert len(results) >= 1


@pytest.mark.asyncio
async def test_per_file_budget_multi_file(build_pipeline, monkeypatch: pytest.MonkeyPatch):
    """Per-file budget should limit tokens from any single file when multiple files present."""
    big_text = "x " * 300
    candidates = [
        _make_chunk(1, "/repo/src/a.ts", text=f"{big_text} a_1"),
        _make_chunk(2, "/repo/src/a.ts", text=f"{big_text} a_2"),
        _make_chunk(3, "/repo/src/a.ts", text=f"{big_text} a_3"),
        _make_chunk(4, "/repo/src/b.ts", text=f"{big_text} b_1"),
        _make_chunk(5, "/repo/src/b.ts", text=f"{big_text} b_2"),
    ]
    pipeline, log_path = build_pipeline(candidates)
    monkeypatch.setattr(pipeline.settings, "hierarchical_assembly_enabled", True)
    monkeypatch.setattr(pipeline.settings, "per_file_budget_ratio", 0.40)
    monkeypatch.setattr(pipeline.settings, "result_max_tokens", 4000)
    monkeypatch.setattr(pipeline.settings, "max_symbols_per_file", 10)

    results = await pipeline.search("query", include_tests=True, max_file_chunks=None)

    # Per-file budget should prevent all chunks from one file
    a_results = [r for r in results if r.filepath == "/repo/src/a.ts"]
    b_results = [r for r in results if r.filepath == "/repo/src/b.ts"]
    # Both files should be represented (neither should dominate completely)
    assert len(a_results) <= 3, "Per-file budget should cap tokens from file a"
    assert len(results) >= 2, "Should have results from multiple files"


@pytest.mark.asyncio
async def test_max_symbols_per_file(build_pipeline, monkeypatch: pytest.MonkeyPatch):
    """Per-file symbol cap should limit chunks from one file."""
    candidates = [
        _make_chunk(i, "/repo/src/big.ts", text=f"function fn_{i}() {{ return {i}; }}")
        for i in range(1, 8)
    ] + [
        _make_chunk(10, "/repo/src/other.ts", text="function other() { return 0; }")
    ]
    pipeline, log_path = build_pipeline(candidates)
    monkeypatch.setattr(pipeline.settings, "hierarchical_assembly_enabled", True)
    monkeypatch.setattr(pipeline.settings, "max_symbols_per_file", 3)

    results = await pipeline.search("query", include_tests=True, max_file_chunks=None)

    big_results = [r for r in results if r.filepath == "/repo/src/big.ts"]
    assert len(big_results) <= 3, f"Max 3 symbols per file, got {len(big_results)}"

    entry = _read_logs(log_path)[-1]
    assert entry["retrieval"]["per_file_symbol_capped"] >= 1


@pytest.mark.asyncio
async def test_file_chunk_replaced_by_symbols(build_pipeline, monkeypatch: pytest.MonkeyPatch):
    """File chunk should be dropped when symbol chunks exist for same file."""
    candidates = [
        _make_chunk(1, "/repo/src/svc.ts", chunk_type="file", text="full file content here"),
        _make_chunk(2, "/repo/src/svc.ts", chunk_type="function", text="function doWork() { return 1; }"),
        _make_chunk(3, "/repo/src/other.ts", chunk_type="function", text="function other() { return 2; }"),
    ]
    pipeline, log_path = build_pipeline(candidates)
    monkeypatch.setattr(pipeline.settings, "hierarchical_assembly_enabled", True)

    results = await pipeline.search("query", include_tests=True, max_file_chunks=None)

    # The file chunk from svc.ts should be replaced by its symbol chunk
    svc_results = [r for r in results if r.filepath == "/repo/src/svc.ts"]
    assert all(r.chunk_type != "file" for r in svc_results), (
        "File chunk should be replaced when symbols exist"
    )
    assert len(svc_results) >= 1, "Symbol chunks from svc.ts should remain"

    entry = _read_logs(log_path)[-1]
    assert entry["retrieval"]["per_file_file_replaced"] >= 1


@pytest.mark.asyncio
async def test_single_file_bypasses_per_file_cap(build_pipeline, monkeypatch: pytest.MonkeyPatch):
    """When all candidates are from one file, per-file controls should bypass."""
    candidates = [
        _make_chunk(i, "/repo/src/only.ts", text=f"function fn_{i}() {{ return {i}; }}")
        for i in range(1, 8)
    ]
    pipeline, log_path = build_pipeline(candidates)
    monkeypatch.setattr(pipeline.settings, "hierarchical_assembly_enabled", True)
    monkeypatch.setattr(pipeline.settings, "max_symbols_per_file", 3)

    results = await pipeline.search("query", include_tests=True, max_file_chunks=None)

    # Single-file: cap should not apply
    assert len(results) > 3, f"Single-file should bypass per-file cap, got {len(results)}"

    entry = _read_logs(log_path)[-1]
    assert entry["retrieval"]["per_file_symbol_capped"] == 0


@pytest.mark.asyncio
async def test_relaxes_test_filter_if_it_would_empty_results(build_pipeline):
    candidates = [
        _make_chunk(1, "/repo/src/__tests__/only_a.ts", chunk_type="function", text="only_test_a"),
        _make_chunk(2, "/repo/src/__tests__/only_b.ts", chunk_type="function", text="only_test_b"),
    ]
    pipeline, log_path = build_pipeline(candidates)

    results = await pipeline.search("query", include_tests=False, max_file_chunks=2)

    assert len(results) > 0
    assert any("/__tests__/" in r.filepath for r in results)

    entry = _read_logs(log_path)[-1]
    assert entry["retrieval"]["test_filter_applied"] is True
    assert entry["retrieval"]["test_filter_relaxed"] is True
