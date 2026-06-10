from __future__ import annotations

import json
import os
import urllib.request

from rag.models import SearchResult


# Princip: Jina reranker je multilingual cross-encoder cez API.
# Je vhodny pre slovencinu a ine jazyky, pretoze hodnoti vztah otazky a chunku
# spolocne, nie iba cez samostatne embeddingy.
# Vhodne pouzitie: velmi dobra API volba pre slovenske a viacjazycne dokumenty,
# ked nechces lokalne tahat vacsi BGE model.
class JinaReranker:
    def __init__(self, model: str = "jina-reranker-v2-base-multilingual"):
        self.model = model
        self.api_key = os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise RuntimeError("JINA_API_KEY is missing for Jina reranking.")

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        payload = {
            "model": self.model,
            "query": query,
            "documents": [result.text for result in results],
            "top_n": top_k or len(results),
        }
        request = urllib.request.Request(
            "https://api.jina.ai/v1/rerank",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))

        reranked = []
        for item in data.get("results", []):
            result = results[item["index"]]
            reranked.append(SearchResult(
                chunk_id=result.chunk_id,
                text=result.text,
                score=float(item["relevance_score"]),
                metadata={
                    **result.metadata,
                    "pre_rerank_score": result.score,
                    "reranker": "jina",
                    "reranker_model": self.model,
                },
            ))

        return reranked
