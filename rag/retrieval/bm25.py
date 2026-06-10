import math
import re
import unicodedata
from collections import Counter

from rag.models import Chunk, SearchResult


TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)

SLOVAK_SUFFIXES = (
    "ami", "ami", "och", "eho", "emu", "ymi", "ymi", "ych", "ych",
    "ovi", "ova", "ove", "ovu", "ami", "ami", "ého", "ému", "ými", "ých",
    "ia", "ie", "iu", "ou", "om", "ov", "mi", "ho", "mu", "ej", "ý", "á",
    "é", "í", "ú", "u", "a", "e", "i", "y",
)

SYNONYMS = {
    "chyba": {"incident", "vada", "porucha", "problem", "nedostupnost"},
    "kriticky": {"priorita", "p1", "vypadok", "kriticka"},
    "pokuta": {"sankcia", "penalizacia", "kompenzacia", "zlava"},
    "oneskorenie": {"omeskanie", "nedodrzanie", "nestihnutie"},
    "vyriesit": {"odstranit", "obnovit", "riesit"},
    "odstavka": {"udrzba", "okno", "nedostupnost"},
    "heslo": {"pristup", "autentizacia", "udaje"},
    "reset": {"deaktivovat", "exspiracia", "obmena"},
    "stavba": {"dielo", "rekonstrukcia", "vystavba"},
    "rozpocet": {"cena", "navysenie", "suma"},
    "termin": {"lehota", "harmonogram", "odovzdanie", "dokoncit"},
}


def normalize_token(token: str) -> str:
    normalized = unicodedata.normalize("NFKD", token.lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def stem_token(token: str) -> str:
    if len(token) <= 4:
        return token

    for suffix in SLOVAK_SUFFIXES:
        if token.endswith(normalize_token(suffix)) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]

    return token


def analyze_text(text: str) -> list[str]:
    tokens = []
    for match in TOKEN_PATTERN.finditer(text):
        token = stem_token(normalize_token(match.group(0)))
        if token:
            tokens.append(token)
    return tokens


def expand_query_terms(terms: list[str]) -> list[str]:
    expanded = list(terms)

    for term in terms:
        for synonym_key, synonyms in SYNONYMS.items():
            normalized_key = stem_token(normalize_token(synonym_key))
            normalized_synonyms = {
                stem_token(normalize_token(synonym))
                for synonym in synonyms
            }

            if term == normalized_key or term in normalized_synonyms:
                expanded.append(normalized_key)
                expanded.extend(normalized_synonyms)

    return expanded


class BM25Retriever:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self.chunk_terms: list[list[str]] = []
        self.doc_lengths: list[int] = []
        self.doc_freqs: dict[str, int] = {}
        self.avg_doc_length = 1.0

    def index(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.chunk_terms = []
        self.doc_lengths = []
        self.doc_freqs = {}

        for chunk in chunks:
            terms = analyze_text(chunk.text)
            self.chunk_terms.append(terms)
            self.doc_lengths.append(len(terms))

            for term in set(terms):
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

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
        query_terms = expand_query_terms(analyze_text(query))
        counts = Counter(self.chunk_terms[chunk_idx])
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
