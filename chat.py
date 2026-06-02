from __future__ import annotations

import argparse
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_COLLECTION = "rag_documents"
DEFAULT_CHROMA_DIR = "chroma_db"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_LLM_MODEL = "gpt-4o-mini"


def resolve_project_path(path: str) -> Path:
    parsed = Path(path)
    if parsed.is_absolute():
        return parsed
    return PROJECT_DIR / parsed


def embed_query(client: OpenAI, question: str, model: str) -> list[float]:
    response = client.embeddings.create(model=model, input=question)
    return response.data[0].embedding


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
    openai_client: OpenAI,
    question: str,
    context: str,
    llm_model: str,
) -> str:
    response = openai_client.responses.create(
        model=llm_model,
        input=[
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
    return response.output_text


def query_rag(
    question: str,
    chroma_dir: Path,
    collection_name: str,
    embedding_model: str,
    llm_model: str,
    top_k: int,
) -> None:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env or your environment.")

    openai_client = OpenAI()
    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = chroma_client.get_collection(collection_name)

    query_embedding = embed_query(openai_client, question, embedding_model)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not documents:
        print("No matching chunks found.")
        return

    context = format_context(documents, metadatas, distances)
    answer = answer_question(openai_client, question, context, llm_model)

    print("\nAnswer")
    print("-" * 60)
    print(answer)

    print("\nRetrieved chunks")
    print("-" * 60)
    for i, (metadata, distance) in enumerate(zip(metadatas, distances), start=1):
        source = metadata.get("source", "unknown")
        chunk_index = metadata.get("chunk_index", "?")
        print(f"{i}. {source} / chunk {chunk_index} / distance {distance:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask questions over the local Chroma RAG index.")
    parser.add_argument("question", nargs="*", help="Question to ask. If omitted, interactive mode starts.")
    parser.add_argument("--chroma-dir", default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
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
            llm_model=args.llm_model,
            top_k=args.top_k,
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
            llm_model=args.llm_model,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
