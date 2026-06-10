from __future__ import annotations

from rag.models import SearchResult


# Princip: tento reranking nerobi ziadne dalsie preusporiadanie vysledkov.
# Zachova poradie, ktore prislo z hybrid search/RRF, a iba oreze zoznam na top_k.
# Vhodne pouzitie: baseline test, ked chces vidiet cisty hybrid search bez
# akekolvek dodatocnej zmeny poradia.
class NoOpReranker:
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        return results[:top_k] if top_k is not None else results
