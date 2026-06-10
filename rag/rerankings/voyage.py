from __future__ import annotations

import json
import os
import urllib.request

from rag.models import SearchResult


# Princip: Voyage reranker je externy API reranker zamerany na nizku latenciu.
# Hybrid search najprv najde kandidatov a Voyage ich nasledne zoradi podla
# relevance otazka-chunk.
# Vhodne pouzitie: ked chces API reranking s dorazom na nizsiu latenciu.
# Oplati sa testovat pri interaktivnom UI, kde je rychlost odpovede dolezita.
class VoyageReranker:
    def __init__(self, model: str = "rerank-2.5"):
        self.model = model
        self.api_key = os.getenv("VOYAGE_API_KEY")
        if not self.api_key:
            raise RuntimeError("VOYAGE_API_KEY is missing for Voyage reranking.")

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
            "top_k": top_k or len(results),
        }
        request = urllib.request.Request(
            "https://api.voyageai.com/v1/rerank",
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
        for item in data.get("data", []):
            result = results[item["index"]]
            reranked.append(SearchResult(
                chunk_id=result.chunk_id,
                text=result.text,
                score=float(item["relevance_score"]),
                metadata={
                    **result.metadata,
                    "pre_rerank_score": result.score,
                    "reranker": "voyage",
                    "reranker_model": self.model,
                },
            ))

        return reranked
