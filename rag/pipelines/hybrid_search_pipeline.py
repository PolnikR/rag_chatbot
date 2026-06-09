from __future__ import annotations

from rag.models import SearchResult
from rag.retrieval.bm25 import BM25Retriever
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.reranking import SimpleReranker
from rag.retrieval.vector import (
    ChromaVectorRetriever,
    OpenAIEmbeddingClient,
    load_chunks_from_chroma,
)


class HybridSearchPipeline:
    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: SimpleReranker,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker

    @classmethod
    def from_chroma(
        cls,
        collection,
        embedding_client: OpenAIEmbeddingClient,
        fusion_k: int = 60,
    ) -> "HybridSearchPipeline":
        chunks = load_chunks_from_chroma(collection)

        bm25_retriever = BM25Retriever()
        bm25_retriever.index(chunks)

        vector_retriever = ChromaVectorRetriever(
            collection=collection,
            embedding_client=embedding_client,
        )

        hybrid_retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            fusion_k=fusion_k,
        )

        return cls(
            hybrid_retriever=hybrid_retriever,
            reranker=SimpleReranker(),
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        candidates = self.hybrid_retriever.retrieve(
            query=query,
            top_k=candidate_k,
            candidate_k=candidate_k,
            filters=filters,
        )

        return self.reranker.rerank(
            query=query,
            results=candidates,
            top_k=top_k,
        )
