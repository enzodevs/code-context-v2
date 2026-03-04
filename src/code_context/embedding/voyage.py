"""Voyage AI client for embeddings and reranking."""

import asyncio
import logging
import random
import time
from typing import Any, Callable, Literal

import tiktoken
import voyageai

from code_context.config import get_settings

logger = logging.getLogger(__name__)

# Tokenizer for estimating batch sizes
_encoder = tiktoken.get_encoding("cl100k_base")


def _estimate_tokens(text: str) -> int:
    """Estimate token count for a text."""
    return len(_encoder.encode(text))


class VoyageClient:
    """Client for Voyage AI embeddings and reranking.

    Uses asymmetric retrieval pattern:
    - voyage-4-large for document indexing (high accuracy)
    - voyage-4-lite for query embedding (shared space, low latency)
    - rerank-2.5 for reranking (high precision for code)
    """

    def __init__(self):
        settings = get_settings()
        self.client = voyageai.Client(api_key=settings.voyage_api_key)
        self.model_index = settings.embedding_model_index
        self.model_query = settings.embedding_model_query
        self.rerank_model = settings.rerank_model
        self.dimensions = settings.embedding_dimensions
        self.batch_max_tokens = settings.embedding_batch_max_tokens
        self.batch_concurrency = settings.embedding_batch_concurrency
        self.max_in_flight_requests = max(1, settings.voyage_max_in_flight_requests)
        self.max_requests_per_minute = max(1, settings.voyage_max_requests_per_minute)
        self.retry_max_attempts = max(1, settings.voyage_retry_max_attempts)
        self.retry_base_delay_ms = max(1, settings.voyage_retry_base_delay_ms)
        self.retry_max_delay_ms = max(self.retry_base_delay_ms, settings.voyage_retry_max_delay_ms)
        self.retry_jitter_ms = max(0, settings.voyage_retry_jitter_ms)

        self._request_sem = asyncio.Semaphore(self.max_in_flight_requests)
        self._rate_lock = asyncio.Lock()
        self._next_request_ts = 0.0
        self._min_request_interval = 60.0 / float(self.max_requests_per_minute)

    async def _throttle_request(self) -> None:
        """Global in-process pacing guardrail for Voyage requests.

        Assigns each request a time slot, then sleeps outside the lock
        so other requests can schedule concurrently.
        """
        async with self._rate_lock:
            now = time.monotonic()
            delay = self._next_request_ts - now
            self._next_request_ts = max(now, self._next_request_ts) + self._min_request_interval
        if delay > 0:
            await asyncio.sleep(delay)

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        """Best-effort retry classification without SDK-specific coupling."""
        raw = str(exc).lower()
        retry_markers = (
            "429",
            "rate limit",
            "too many requests",
            "timeout",
            "temporarily unavailable",
            "service unavailable",
            "connection reset",
            "connection aborted",
            "connection refused",
            "server error",
            "502",
            "503",
            "504",
        )
        if any(marker in raw for marker in retry_markers):
            return True

        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int):
            return status_code == 429 or 500 <= status_code <= 504

        return False

    async def _call_with_retry(
        self,
        label: str,
        fn: Callable[..., Any],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute Voyage SDK call with pacing + bounded retries."""
        attempt = 0
        while True:
            attempt += 1
            try:
                async with self._request_sem:
                    await self._throttle_request()
                    return await asyncio.to_thread(fn, *args, **kwargs)
            except Exception as e:
                is_retryable = self._is_retryable_error(e)
                if attempt >= self.retry_max_attempts or not is_retryable:
                    logger.error(f"{label} error (attempt {attempt}/{self.retry_max_attempts}): {e}")
                    raise

                backoff_ms = min(
                    self.retry_max_delay_ms,
                    self.retry_base_delay_ms * (2 ** (attempt - 1)),
                )
                jitter_ms = random.randint(0, self.retry_jitter_ms) if self.retry_jitter_ms else 0
                sleep_ms = backoff_ms + jitter_ms
                logger.warning(
                    f"{label} retry {attempt}/{self.retry_max_attempts} after {sleep_ms}ms: {e}"
                )
                await asyncio.sleep(sleep_ms / 1000.0)

    async def embed_documents(
        self,
        texts: list[str],
        input_type: Literal["document", "query"] = "document",
    ) -> list[list[float]]:
        """Embed documents for indexing.

        Uses voyage-4-large for document embeddings with high accuracy.
        Batches by token count to respect API limit (120k tokens).
        """
        if not texts:
            return []

        model = self.model_index if input_type == "document" else self.model_query
        all_embeddings: list[list[float]] = []

        # Build batches by token count
        batches: list[list[str]] = []
        current_batch: list[str] = []
        current_tokens = 0

        for text in texts:
            text_tokens = _estimate_tokens(text)

            # If single text exceeds limit, it will be sent alone (API will truncate)
            if text_tokens > self.batch_max_tokens:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0
                batches.append([text])
                logger.warning(f"Single chunk exceeds token limit: {text_tokens} tokens")
                continue

            # If adding this text exceeds limit, start new batch
            if current_tokens + text_tokens > self.batch_max_tokens:
                batches.append(current_batch)
                current_batch = [text]
                current_tokens = text_tokens
            else:
                current_batch.append(text)
                current_tokens += text_tokens

        if current_batch:
            batches.append(current_batch)

        # Process batches concurrently (respecting semaphore limit)
        sem = asyncio.Semaphore(self.batch_concurrency)

        async def _embed_batch(idx: int, batch: list[str]) -> tuple[int, list[list[float]]]:
            async with sem:
                result = await self._call_with_retry(
                    "Embedding",
                    self.client.embed,
                    texts=batch,
                    model=model,
                    input_type=input_type,
                    output_dimension=self.dimensions,
                    output_dtype="float",
                )
                logger.debug(
                    f"Embedded batch {idx + 1}/{len(batches)}, "
                    f"chunks: {len(batch)}, tokens: {result.total_tokens}"
                )
                return idx, result.embeddings

        try:
            results = await asyncio.gather(
                *[_embed_batch(i, b) for i, b in enumerate(batches)]
            )
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            raise

        for _, embeddings in sorted(results, key=lambda x: x[0]):
            all_embeddings.extend(embeddings)

        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query for search.

        Uses voyage-4-lite for fast, low-latency query embeddings.
        Compatible with voyage-4-large document embeddings (shared space).
        """
        try:
            result = await self._call_with_retry(
                "Query embedding",
                self.client.embed,
                texts=[query],
                model=self.model_query,
                input_type="query",
                output_dimension=self.dimensions,
                output_dtype="float",
            )
            return result.embeddings[0]
        except Exception as e:
            logger.error(f"Query embedding error: {e}")
            raise

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """Rerank documents by relevance to query.

        Uses rerank-2.5 for high precision code reranking.

        Returns:
            List of (original_index, relevance_score) sorted by score descending.
        """
        if not documents:
            return []

        settings = get_settings()
        top_k = top_k or settings.rerank_top_k_output

        try:
            result = await self._call_with_retry(
                "Reranking",
                self.client.rerank,
                query=query,
                documents=documents,
                model=self.rerank_model,
                top_k=min(top_k, len(documents)),
            )

            return [
                (r.index, r.relevance_score)
                for r in result.results
            ]
        except Exception as e:
            logger.error(f"Reranking error: {e}")
            raise


# Singleton instance
_client: VoyageClient | None = None


def get_voyage_client() -> VoyageClient:
    """Get or create the Voyage client singleton."""
    global _client
    if _client is None:
        _client = VoyageClient()
    return _client
