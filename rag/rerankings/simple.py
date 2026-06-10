from __future__ import annotations

from rag.models import SearchResult
from rag.retrieval.bm25 import analyze_text, expand_query_terms


STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what", "how", "why",
    "when", "where", "do", "does", "for", "of", "in", "to", "and", "or",
    "on", "at", "by", "it", "its", "this", "that", "with", "from", "be",
    "has", "have", "had", "not", "but",
    "a", "ak", "aka", "ake", "aky", "ako", "alebo", "aj", "bez", "bol",
    "bola", "boli", "bolo", "by", "cez", "co", "do", "je", "jeho", "jej",
    "ked", "kde", "ktora", "ktore", "ktory", "ma", "na", "nad", "nie",
    "od", "pod", "podla", "po", "pre", "pri", "sa", "si", "su", "ta",
    "tak", "te", "ten", "to", "v", "vo", "za", "ze",
}


# Princip: jednoduchy lokalny reranking bez dalsieho modelu.
# Kandidatov zoradi podla prekryvu slov z otazky, dvojic slov a toho, ci sa
# dolezite slova nachadzaju skor v texte. Je rychly, ale nechape vyznam tak
# dobre ako cross-encoder.
# Vhodne pouzitie: lokalny prototyp, male datasety a rychle porovnanie bez API
# klucov. Pre slovencinu je pouzitelny ako lacny baseline, nie ako najlepsia
# presnost.
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
            text_terms = self._content_terms(result.text)

            term_overlap = len(query_terms & text_terms)
            bigram_matches = len(query_bigrams & self._bigrams(result.text))
            position_boost = self._position_boost(query_terms, result.text)

            rerank_score = (
                result.score * 20.0
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
                    "reranker": "simple",
                    "term_overlap": term_overlap,
                    "bigram_matches": bigram_matches,
                    "position_boost": position_boost,
                },
            ))

        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:top_k] if top_k is not None else reranked

    def _content_terms(self, text: str) -> set[str]:
        expanded_terms = expand_query_terms(analyze_text(text))
        return {
            token
            for token in expanded_terms
            if token not in STOP_WORDS
        }

    def _bigrams(self, text: str) -> set[tuple[str, str]]:
        tokens = [
            token
            for token in analyze_text(text)
            if token not in STOP_WORDS
        ]
        return {
            (tokens[index], tokens[index + 1])
            for index in range(len(tokens) - 1)
        }

    def _position_boost(self, query_terms: set[str], text: str) -> float:
        score = 0.0
        text_tokens = analyze_text(text)
        early_cutoff = max(1, len(text_tokens) // 3)

        for term in query_terms:
            try:
                position = text_tokens.index(term)
            except ValueError:
                continue

            if position < early_cutoff:
                score += 0.5

        return score
