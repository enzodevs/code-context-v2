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
