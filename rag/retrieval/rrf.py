from rag.models import SearchResult


def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchResult]],
    k: int = 60,
) -> list[SearchResult]:
    scores: dict[str, float] = {}
    by_id: dict[str, SearchResult] = {}

    for results in ranked_lists:
        for rank, result in enumerate(results, start=1):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (k + rank)
            by_id[result.chunk_id] = result

    fused = []
    for chunk_id, score in scores.items():
        original = by_id[chunk_id]
        fused.append(SearchResult(
            chunk_id=original.chunk_id,
            text=original.text,
            score=score,
            metadata=original.metadata,
        ))

    return sorted(fused, key=lambda result: result.score, reverse=True)