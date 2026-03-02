"""Unit tests for rerank intent prompt composition."""

from code_context.retrieval.pipeline import RetrievalPipeline


def test_resolve_search_intent_defaults_to_implementation():
    assert RetrievalPipeline._resolve_search_intent(None) == "implementation"
    assert RetrievalPipeline._resolve_search_intent("") == "implementation"
    assert RetrievalPipeline._resolve_search_intent("unknown") == "implementation"


def test_resolve_search_intent_supports_aliases():
    assert RetrievalPipeline._resolve_search_intent("impl") == "implementation"
    assert RetrievalPipeline._resolve_search_intent("perf") == "performance"
    assert RetrievalPipeline._resolve_search_intent("authz") == "security"
    assert RetrievalPipeline._resolve_search_intent("arch") == "architecture"


def test_build_rerank_query_uses_definition_instruction():
    query = "type definitions for api responses"
    rerank_query = RetrievalPipeline._build_rerank_query(query, "definition")

    assert "Prioritize canonical declarations" in rerank_query
    assert "Ranking rules:" in rerank_query
    assert f"Query: {query}" in rerank_query


def test_build_rerank_query_deprioritizes_tests_by_default():
    query = "authorization middleware for admin actions"
    rerank_query = RetrievalPipeline._build_rerank_query(query, "security")

    assert "Deprioritize test/spec/fixture/mocked code" in rerank_query


def test_build_rerank_query_keeps_test_queries_without_penalty():
    query = "integration test for refresh token rotation"
    rerank_query = RetrievalPipeline._build_rerank_query(query, "usage")

    assert "Deprioritize test/spec/fixture/mocked code" not in rerank_query


def test_build_rerank_query_avoids_false_positive_for_latest_word():
    query = "latest auth token rotation flow implementation"
    rerank_query = RetrievalPipeline._build_rerank_query(query, "implementation")

    assert "Deprioritize test/spec/fixture/mocked code" in rerank_query
