from __future__ import annotations

from importlib import import_module

from rag.models import SearchResult


# Princip: cross-encoder cita otazku a kandidatny chunk naraz.
# Vdaka tomu vie lepsie posudit, ci chunk skutocne odpoveda na otazku, ale je
# pomalsi a vyzaduje lokalny model zo sentence-transformers.
# Vhodne pouzitie: lokalne experimenty, ked nechces pouzit externe API.
# Default model je maly a rychly, ale skor anglicky; pre slovenske dokumenty
# preferuj reranker mode "bge".
class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            sentence_transformers = import_module("sentence_transformers")
        except ImportError as exc:
            raise RuntimeError(
                "CrossEncoder reranker requires sentence-transformers. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        self.model_name = model_name
        self.model = sentence_transformers.CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        pairs = [(query, result.text) for result in results]
        scores = self.model.predict(pairs)

        reranked = [
            SearchResult(
                chunk_id=result.chunk_id,
                text=result.text,
                score=float(score),
                metadata={
                    **result.metadata,
                    "pre_rerank_score": result.score,
                    "reranker": "cross_encoder",
                    "reranker_model": self.model_name,
                },
            )
            for result, score in zip(results, scores)
        ]
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:top_k] if top_k is not None else reranked
