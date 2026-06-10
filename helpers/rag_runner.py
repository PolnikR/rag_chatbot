from __future__ import annotations

import os
import time
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

from helpers.clients import create_llm_client, embed_query
from helpers.formatting import format_search_results_context
from helpers.llm import answer_question
from helpers.utils import estimate_tokens
from rag.models import SearchResult
from rag.pipelines.hybrid_search_pipeline import HybridSearchPipeline
from rag.retrieval.bm25 import BM25Retriever
from rag.retrieval.vector import OpenAIEmbeddingClient, load_chunks_from_chroma


def retrieve_context(
    question: str,
    collection,
    embedding_client: OpenAI,
    embedding_model: str,
    top_k: int,
    retrieval_mode: str,
    candidate_k: int,
    reranker_mode: str = "simple",
) -> tuple[str, list[SearchResult], int, float]:
    retrieval_start = time.perf_counter()
    query_embedding_tokens = 0

    if retrieval_mode == "vector":
        query_embedding, query_embedding_tokens = embed_query(
            embedding_client,
            question,
            embedding_model,
        )
        raw = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        retrieval_results = [
            SearchResult(
                chunk_id=chunk_id,
                text=document,
                score=1.0 - distance,
                metadata={
                    **(metadata or {}),
                    "distance": distance,
                    "retrieval_mode": "vector",
                },
            )
            for chunk_id, document, metadata, distance in zip(
                raw["ids"][0],
                raw["documents"][0],
                raw["metadatas"][0],
                raw["distances"][0],
            )
        ]
    elif retrieval_mode == "bm25":
        chunks = load_chunks_from_chroma(collection)
        bm25_retriever = BM25Retriever()
        bm25_retriever.index(chunks)
        retrieval_results = bm25_retriever.retrieve(question, top_k=top_k)
        retrieval_results = [
            SearchResult(
                chunk_id=result.chunk_id,
                text=result.text,
                score=result.score,
                metadata={
                    **result.metadata,
                    "retrieval_mode": "bm25",
                },
            )
            for result in retrieval_results
        ]
    elif retrieval_mode == "hybrid":
        openai_embedding_client = OpenAIEmbeddingClient(
            client=embedding_client,
            model=embedding_model,
        )
        retrieval_pipeline = HybridSearchPipeline.from_chroma(
            collection=collection,
            embedding_client=openai_embedding_client,
            reranker_mode=reranker_mode,
        )
        retrieval_results = retrieval_pipeline.retrieve(
            query=question,
            top_k=top_k,
            candidate_k=candidate_k,
        )
        retrieval_results = [
            SearchResult(
                chunk_id=result.chunk_id,
                text=result.text,
                score=result.score,
                metadata={
                    **result.metadata,
                    "retrieval_mode": "hybrid",
                    "reranker_mode": reranker_mode,
                },
            )
            for result in retrieval_results
        ]
        query_embedding_tokens = estimate_tokens(question)
    else:
        raise ValueError(f"Unsupported retrieval mode: {retrieval_mode}")

    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
    return (
        format_search_results_context(retrieval_results),
        retrieval_results,
        query_embedding_tokens,
        retrieval_ms,
    )


def run_rag(
    question: str,
    chroma_dir: Path,
    collection_name: str,
    embedding_model: str,
    llm_provider: str,
    llm_model: str,
    top_k: int,
    retrieval_mode: str,
    candidate_k: int,
    reranker_mode: str = "simple",
) -> dict:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env or your environment.")

    embedding_client = OpenAI()
    llm_client = create_llm_client(llm_provider)
    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = chroma_client.get_collection(collection_name)

    context, retrieval_results, query_embedding_tokens, retrieval_ms = retrieve_context(
        question=question,
        collection=collection,
        embedding_client=embedding_client,
        embedding_model=embedding_model,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
        candidate_k=candidate_k,
        reranker_mode=reranker_mode,
    )

    if not retrieval_results:
        return {
            "answer": "No matching chunks found.",
            "context": "",
            "retrieval_results": [],
            "stats": {
                "embed_ms": 0.0,
                "retrieval_ms": retrieval_ms,
                "llm_ms": 0.0,
                "query_embedding_tokens": query_embedding_tokens,
                "llm_input_tokens": 0,
                "llm_output_tokens": 0,
            },
        }

    llm_start = time.perf_counter()
    answer, llm_input_tokens, llm_output_tokens = answer_question(
        llm_client,
        question,
        context,
        llm_model,
    )
    llm_ms = (time.perf_counter() - llm_start) * 1000

    return {
        "answer": answer,
        "context": context,
        "retrieval_results": retrieval_results,
        "stats": {
            "embed_ms": 0.0,
            "retrieval_ms": retrieval_ms,
            "llm_ms": llm_ms,
            "query_embedding_tokens": query_embedding_tokens,
            "llm_input_tokens": llm_input_tokens,
            "llm_output_tokens": llm_output_tokens,
        },
    }
