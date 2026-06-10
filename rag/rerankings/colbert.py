from __future__ import annotations

from rag.models import SearchResult


# Princip: ColBERT pouziva multi-vector reprezentaciu, kde sa porovnavaju
# tokeny otazky s tokenmi dokumentu. Vie byt presnejsi ako jeden embedding pre
# cely chunk, ale vyzaduje specialny index a samostatnu integraciu.
# Vhodne pouzitie: pokrocile RAG riesenia s velkym korpusom, ked si ochotny
# budovat samostatny ColBERT index. Pre tento maly projekt ho zatial nepouzivaj.
class ColbertReranker:
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        raise NotImplementedError(
            "ColBERT reranking requires a ColBERT index and is not wired yet."
        )
