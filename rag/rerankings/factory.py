from __future__ import annotations

from rag.rerankings.base import BaseReranker
from rag.rerankings.bge import BgeReranker
from rag.rerankings.cohere import CohereReranker
from rag.rerankings.colbert import ColbertReranker
from rag.rerankings.cross_encoder import CrossEncoderReranker
from rag.rerankings.jina import JinaReranker
from rag.rerankings.none import NoOpReranker
from rag.rerankings.simple import SimpleReranker
from rag.rerankings.voyage import VoyageReranker


def create_reranker(mode: str) -> BaseReranker:
    normalized_mode = mode.strip().lower().replace("-", "_")

    if normalized_mode == "none":
        return NoOpReranker()
    if normalized_mode == "simple":
        return SimpleReranker()
    if normalized_mode == "cross_encoder":
        return CrossEncoderReranker()
    if normalized_mode == "cohere":
        return CohereReranker()
    if normalized_mode == "voyage":
        return VoyageReranker()
    if normalized_mode == "jina":
        return JinaReranker()
    if normalized_mode == "bge":
        return BgeReranker()
    if normalized_mode == "colbert":
        return ColbertReranker()

    raise ValueError(f"Unsupported reranker mode: {mode}")
