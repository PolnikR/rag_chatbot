from rag.models import SearchResult
from rag.retrieval.rrf import reciprocal_rank_fusion


class HybridRetriever:
    def __init__(self, vector_retriever, bm25_retriever, fusion_k: int = 60):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.fusion_k = fusion_k

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        vector_results = self.vector_retriever.retrieve(
            query=query,
            top_k=candidate_k,
            filters=filters,
        )

        bm25_results = self.bm25_retriever.retrieve(
            query=query,
            top_k=candidate_k,
        )

        fused = reciprocal_rank_fusion(
            [vector_results, bm25_results],
            k=self.fusion_k,
        )

        return fused[:top_k]