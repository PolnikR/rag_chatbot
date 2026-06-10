from __future__ import annotations

from typing import Protocol

from rag.models import SearchResult


class BaseReranker(Protocol):
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        ...
