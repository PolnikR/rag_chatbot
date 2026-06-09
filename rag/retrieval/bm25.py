import math
from collections import Counter

from rag.models import Chunk, SearchResult


class BM25Retriever:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self.doc_lengths: list[int] = []
        self.doc_freqs: dict[str, int] = {}
        self.avg_doc_length = 1.0

    def index(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.doc_lengths = []
        self.doc_freqs = {}

        for chunk in chunks:
            words = chunk.text.lower().split()
            self.doc_lengths.append(len(words))

            for word in set(words):
                self.doc_freqs[word] = self.doc_freqs.get(word, 0) + 1

        if self.doc_lengths:
            self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)

    def retrieve(self, query: str, top_k: int = 10) -> list[SearchResult]:
        scored = []

        for idx, chunk in enumerate(self.chunks):
            score = self._score(query, idx)
            scored.append(SearchResult(
                chunk_id=chunk.id,
                text=chunk.text,
                score=score,
                metadata=chunk.metadata,
            ))

        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:top_k]

    def _score(self, query: str, chunk_idx: int) -> float:
        query_terms = query.lower().split()
        chunk_words = self.chunks[chunk_idx].text.lower().split()
        counts = Counter(chunk_words)
        doc_len = self.doc_lengths[chunk_idx]
        total_docs = len(self.chunks)

        score = 0.0

        for term in query_terms:
            if term not in counts:
                continue

            tf = counts[term]
            df = self.doc_freqs.get(term, 0)
            idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1)

            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * doc_len / self.avg_doc_length
            )

            score += idf * numerator / denominator

        return score