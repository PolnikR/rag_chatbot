from __future__ import annotations

import json
import os
import urllib.request

from rag.models import SearchResult


# Princip: Cohere Rerank je externy cross-encoder reranker cez API.
# Posle otazku a kandidatne chunky do Cohere, ktore vrati relevance score pre
# kazdy chunk. Je presnejsi ako heuristika, ale zavisi od siete a API kluca.
# Vhodne pouzitie: ked chces kvalitny managed reranking bez lokalneho modelu a
# nevadi ti externa API sluzba. Dobry na zmiesane datasety a produkcne testy.
class CohereReranker:
    def __init__(self, model: str = "rerank-v3.5"):
        self.model = model
        self.api_key = os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise RuntimeError("COHERE_API_KEY is missing for Cohere reranking.")

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
            "https://api.cohere.com/v2/rerank",
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
                    "reranker": "cohere",
                    "reranker_model": self.model,
                },
            ))

        return reranked
