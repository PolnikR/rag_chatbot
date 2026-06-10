from __future__ import annotations

from rag.rerankings.cross_encoder import CrossEncoderReranker


# Princip: BGE reranker je open-source cross-encoder model.
# Bezi lokalne cez sentence-transformers, cita otazku a chunk naraz a je vhodny
# ako silnejsia alternativa k jednoduchemu overlap rerankingu.
# Vhodne pouzitie: najlepsia lokalna volba pre slovencinu a multilingual texty
# v tomto projekte. Nepotrebuje API kluc, ale model je vacsi a pomalsi ako
# default cross_encoder.
class BgeReranker(CrossEncoderReranker):
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        super().__init__(model_name=model_name)
