from __future__ import annotations

import os

from helpers.config import (
    EMBEDDING_PRICE_PER_1M_TOKENS,
    GPT_4O_MINI_INPUT_PRICE_PER_1M_TOKENS,
    GPT_4O_MINI_OUTPUT_PRICE_PER_1M_TOKENS,
)
from helpers.utils import cost_for_tokens


def format_search_results_context(results) -> str:
    blocks = []
    for i, result in enumerate(results, start=1):
        metadata = result.metadata
        source = metadata.get("source", "unknown")
        chunk_index = metadata.get("chunk_index", "?")
        distance = metadata.get("distance")
        pre_rerank_score = metadata.get("pre_rerank_score")
        distance_text = f", distance {distance:.4f}" if isinstance(distance, float) else ""
        rerank_text = (
            f", pre-rerank {pre_rerank_score:.4f}"
            if isinstance(pre_rerank_score, float)
            else ""
        )
        blocks.append(
            f"[Source {i}: {source}, chunk {chunk_index}{distance_text}{rerank_text}]\n"
            f"{result.text}"
        )
    return "\n\n---\n\n".join(blocks)


def print_stats(
    embed_ms: float,
    vector_ms: float,
    llm_ms: float,
    query_embedding_tokens: int,
    llm_input_tokens: int,
    llm_output_tokens: int,
    llm_provider: str,
) -> None:
    embedding_cost = cost_for_tokens(query_embedding_tokens, EMBEDDING_PRICE_PER_1M_TOKENS)
    if llm_provider == "openai":
        input_cost = cost_for_tokens(llm_input_tokens, GPT_4O_MINI_INPUT_PRICE_PER_1M_TOKENS)
        output_cost = cost_for_tokens(llm_output_tokens, GPT_4O_MINI_OUTPUT_PRICE_PER_1M_TOKENS)
    else:
        input_price = float(os.getenv("DEEPSEEK_INPUT_PRICE_PER_1M_TOKENS", "0"))
        output_price = float(os.getenv("DEEPSEEK_OUTPUT_PRICE_PER_1M_TOKENS", "0"))
        input_cost = cost_for_tokens(llm_input_tokens, input_price)
        output_cost = cost_for_tokens(llm_output_tokens, output_price)
    total_cost = embedding_cost + input_cost + output_cost

    print("\nStats")
    print("-" * 60)
    print(f"Latency embed query:   {embed_ms:.0f} ms")
    print(f"Latency vector search: {vector_ms:.0f} ms")
    print(f"Latency LLM answer:    {llm_ms:.0f} ms")
    print(f"Latency total:         {(embed_ms + vector_ms + llm_ms):.0f} ms")
    print(f"Query embedding tokens: {query_embedding_tokens}")
    print(f"LLM input tokens:       {llm_input_tokens}")
    print(f"LLM output tokens:      {llm_output_tokens}")
    if llm_provider == "deepseek" and input_cost == 0 and output_cost == 0:
        print(f"Estimated embedding cost/query: ${embedding_cost:.6f}")
        print("Estimated LLM cost: DeepSeek pricing not configured")
    else:
        print(f"Estimated cost/query:   ${total_cost:.6f}")
        print(f"Estimated cost/1000:    ${total_cost * 1000:.2f}")
