from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_COLLECTION = "rag_documents"
DEFAULT_CHROMA_DIR = "chroma_db"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_LLM_PROVIDER = "deepseek"
DEFAULT_OPENAI_LLM_MODEL = "gpt-4o-mini"
DEFAULT_DEEPSEEK_LLM_MODEL = "deepseek-chat"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
EMBEDDING_PRICE_PER_1M_TOKENS = 0.02
GPT_4O_MINI_INPUT_PRICE_PER_1M_TOKENS = 0.15
GPT_4O_MINI_OUTPUT_PRICE_PER_1M_TOKENS = 0.60


def resolve_project_path(path: str) -> Path:
    parsed = Path(path)
    if parsed.is_absolute():
        return parsed
    return PROJECT_DIR / parsed


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


def usage_value(usage: object, name: str, default: int = 0) -> int:
    value = getattr(usage, name, default)
    return int(value or default)


def cost_for_tokens(tokens: int, price_per_1m: float) -> float:
    return (tokens / 1_000_000) * price_per_1m


def embed_query(client: OpenAI, question: str, model: str) -> tuple[list[float], int]:
    response = client.embeddings.create(model=model, input=question)
    usage = getattr(response, "usage", None)
    tokens = usage_value(usage, "total_tokens", estimate_tokens(question))
    return response.data[0].embedding, tokens


def create_llm_client(provider: str) -> OpenAI:
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env or your environment.")
        return OpenAI()

    if provider == "deepseek":
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        if not deepseek_api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is missing. Add it to .env or your environment.")
        return OpenAI(
            api_key=deepseek_api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL),
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")


def format_context(documents: list[str], metadatas: list[dict], distances: list[float]) -> str:
    blocks = []
    for i, (document, metadata, distance) in enumerate(zip(documents, metadatas, distances), start=1):
        source = metadata.get("source", "unknown")
        chunk_index = metadata.get("chunk_index", "?")
        blocks.append(
            f"[Source {i}: {source}, chunk {chunk_index}, distance {distance:.4f}]\n"
            f"{document}"
        )
    return "\n\n---\n\n".join(blocks)


def answer_question(
    llm_client: OpenAI,
    question: str,
    context: str,
    llm_model: str,
) -> tuple[str, int, int]:
    response = llm_client.chat.completions.create(
        model=llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a RAG assistant. Answer only from the provided context. "
                    "If the context does not contain the answer, say: "
                    "'I do not have enough information in the indexed documents.' "
                    "Every factual claim must cite at least one source number like [Source 1]. "
                    "Keep the answer concise."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
    )
    usage = getattr(response, "usage", None)
    answer = response.choices[0].message.content or ""
    input_tokens = usage_value(usage, "prompt_tokens", estimate_tokens(context) + estimate_tokens(question))
    output_tokens = usage_value(usage, "completion_tokens", estimate_tokens(answer))
    return answer, input_tokens, output_tokens


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


def query_rag(
    question: str,
    chroma_dir: Path,
    collection_name: str,
    embedding_model: str,
    llm_provider: str,
    llm_model: str,
    top_k: int,
    show_stats: bool,
) -> None:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env or your environment.")

    embedding_client = OpenAI()
    llm_client = create_llm_client(llm_provider)
    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = chroma_client.get_collection(collection_name)

    embed_start = time.perf_counter()
    query_embedding, query_embedding_tokens = embed_query(embedding_client, question, embedding_model)
    embed_ms = (time.perf_counter() - embed_start) * 1000

    vector_start = time.perf_counter()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    vector_ms = (time.perf_counter() - vector_start) * 1000

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not documents:
        print("No matching chunks found.")
        return

    context = format_context(documents, metadatas, distances)
    llm_start = time.perf_counter()
    answer, llm_input_tokens, llm_output_tokens = answer_question(
        llm_client,
        question,
        context,
        llm_model,
    )
    llm_ms = (time.perf_counter() - llm_start) * 1000

    print("\nAnswer")
    print("-" * 60)
    print(answer)

    if show_stats:
        print_stats(
            embed_ms=embed_ms,
            vector_ms=vector_ms,
            llm_ms=llm_ms,
            query_embedding_tokens=query_embedding_tokens,
            llm_input_tokens=llm_input_tokens,
            llm_output_tokens=llm_output_tokens,
            llm_provider=llm_provider,
        )

    print("\nRetrieved chunks")
    print("-" * 60)
    for i, (metadata, distance) in enumerate(zip(metadatas, distances), start=1):
        source = metadata.get("source", "unknown")
        chunk_index = metadata.get("chunk_index", "?")
        print(f"{i}. {source} / chunk {chunk_index} / distance {distance:.4f}")


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
    parser.add_argument("--no-stats", action="store_true", help="Hide latency and cost stats.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chroma_dir = resolve_project_path(args.chroma_dir)

    if args.question:
        question = " ".join(args.question)
        query_rag(
            question=question,
            chroma_dir=chroma_dir,
            collection_name=args.collection,
            embedding_model=args.embedding_model,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            top_k=args.top_k,
            show_stats=not args.no_stats,
        )
        return

    print("RAG chat. Type 'exit' or 'quit' to stop.")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        query_rag(
            question=question,
            chroma_dir=chroma_dir,
            collection_name=args.collection,
            embedding_model=args.embedding_model,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            top_k=args.top_k,
            show_stats=not args.no_stats,
        )


if __name__ == "__main__":
    main()
