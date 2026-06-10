from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from helpers.config import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_COLLECTION,
    DEFAULT_DEEPSEEK_LLM_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_OPENAI_LLM_MODEL,
)
from helpers.formatting import print_stats
from helpers.rag_runner import run_rag


def query_rag(
    question: str,
    chroma_dir: Path,
    collection_name: str,
    embedding_model: str,
    llm_provider: str,
    llm_model: str,
    top_k: int,
    retrieval_mode: str,
    candidate_k: int,
    reranker_mode: str,
    show_stats: bool,
) -> None:
    result = run_rag(
        question=question,
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
        llm_provider=llm_provider,
        llm_model=llm_model,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
        candidate_k=candidate_k,
        reranker_mode=reranker_mode,
    )

    print("\nAnswer")
    print("-" * 60)
    print(result["answer"])

    if show_stats:
        stats = result["stats"]
        print_stats(
            embed_ms=stats["embed_ms"],
            vector_ms=stats["retrieval_ms"],
            llm_ms=stats["llm_ms"],
            query_embedding_tokens=stats["query_embedding_tokens"],
            llm_input_tokens=stats["llm_input_tokens"],
            llm_output_tokens=stats["llm_output_tokens"],
            llm_provider=llm_provider,
        )

    print("\nRetrieved chunks")
    print("-" * 60)
    for i, search_result in enumerate(result["retrieval_results"], start=1):
        source = search_result.metadata.get("source", "unknown")
        chunk_index = search_result.metadata.get("chunk_index", "?")
        mode = search_result.metadata.get("retrieval_mode", retrieval_mode)
        print(f"{i}. {source} / chunk {chunk_index} / {mode} score {search_result.score:.4f}")


def parse_args() -> argparse.Namespace:
    load_dotenv()
    default_llm_provider = os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER)
    default_llm_model = os.getenv(
        "DEEPSEEK_MODEL" if default_llm_provider == "deepseek" else "OPENAI_LLM_MODEL",
        DEFAULT_DEEPSEEK_LLM_MODEL if default_llm_provider == "deepseek" else DEFAULT_OPENAI_LLM_MODEL,
    )

    parser = argparse.ArgumentParser(description="Ask questions over the local Chroma RAG index.")
    parser.add_argument("question", nargs="*", help="Question to ask. If omitted, interactive mode starts.")
    parser.add_argument("--chroma-dir", default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--llm-provider", choices=["openai", "deepseek"], default=default_llm_provider)
    parser.add_argument("--llm-model", default=default_llm_model)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--retrieval-mode", choices=["vector", "bm25", "hybrid"], default="hybrid")
    parser.add_argument(
        "--reranker-mode",
        choices=["none", "simple", "cross_encoder", "cohere", "voyage", "jina", "bge", "colbert"],
        default="simple",
        help="Reranker used only for hybrid retrieval.",
    )
    parser.add_argument("--no-stats", action="store_true", help="Hide latency and cost stats.")
    return parser.parse_args()
