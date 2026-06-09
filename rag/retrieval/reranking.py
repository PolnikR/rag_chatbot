from __future__ import annotations

import re

from rag.models import SearchResult


STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what", "how", "why",
    "when", "where", "do", "does", "for", "of", "in", "to", "and", "or",
    "on", "at", "by", "it", "its", "this", "that", "with", "from", "be",
    "has", "have", "had", "not", "but",
}

TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


class SimpleReranker:
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        query_terms = self._content_terms(query)
        query_bigrams = self._bigrams(query)

        reranked = []
        for result in results:
            text = result.text.lower()
            text_terms = set(self._tokens(text))

            term_overlap = len(query_terms & text_terms)
            bigram_matches = sum(1 for bigram in query_bigrams if bigram in text)
            position_boost = self._position_boost(query_terms, text)

            rerank_score = (
                result.score * 5.0
                + term_overlap * 1.0
                + bigram_matches * 2.0
                + position_boost
            )

            reranked.append(SearchResult(
                chunk_id=result.chunk_id,
                text=result.text,
                score=rerank_score,
                metadata={
                    **result.metadata,
                    "pre_rerank_score": result.score,
                    "term_overlap": term_overlap,
                    "bigram_matches": bigram_matches,
                    "position_boost": position_boost,
                },
            ))

        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:top_k] if top_k is not None else reranked

    def _content_terms(self, text: str) -> set[str]:
        return {
            token
            for token in self._tokens(text)
            if token not in STOP_WORDS
        }

    def _bigrams(self, text: str) -> set[str]:
        tokens = [
            token
            for token in self._tokens(text)
            if token not in STOP_WORDS
        ]
        return {
            f"{tokens[index]} {tokens[index + 1]}"
            for index in range(len(tokens) - 1)
        }

    def _position_boost(self, query_terms: set[str], text: str) -> float:
        score = 0.0
        early_cutoff = max(1, len(text) // 3)

        for term in query_terms:
            position = text.find(term)
            if 0 <= position < early_cutoff:
                score += 0.5

        return score

    def _tokens(self, text: str) -> list[str]:
        return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]
