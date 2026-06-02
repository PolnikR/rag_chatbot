from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_COLLECTION = "rag_documents"
DEFAULT_DOCUMENTS_DIR = "documents"
DEFAULT_CHROMA_DIR = "chroma_db"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
SUPPORTED_EXTENSIONS = {".txt", ".md"}


def read_documents(documents_dir: Path) -> list[dict[str, str]]:
    documents = []
    for path in sorted(documents_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        documents.append({
            "source": str(path.relative_to(documents_dir)),
            "text": text,
        })

    return documents


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def stable_chunk_id(source: str, chunk_index: int, chunk: str) -> str:
    digest = hashlib.sha256(f"{source}:{chunk_index}:{chunk}".encode("utf-8")).hexdigest()
    return digest[:24]


def embed_texts(client: OpenAI, texts: list[str], model: str, batch_size: int) -> list[list[float]]:
    embeddings: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        embeddings.extend(item.embedding for item in response.data)

    return embeddings


def build_index(
    documents_dir: Path,
    chroma_dir: Path,
    collection_name: str,
    embedding_model: str,
    chunk_size: int,
    overlap: int,
    batch_size: int,
    reset: bool,
) -> None:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env or your environment.")

    documents = read_documents(documents_dir)
    if not documents:
        raise RuntimeError(
            f"No documents found in {documents_dir}. "
            f"Add .txt or .md files before running ingestion."
        )

    ids = []
    chunks = []
    metadatas = []

    for document in documents:
        source = document["source"]
        document_chunks = chunk_text(document["text"], chunk_size=chunk_size, overlap=overlap)

        for chunk_index, chunk in enumerate(document_chunks):
            ids.append(stable_chunk_id(source, chunk_index, chunk))
            chunks.append(chunk)
            metadatas.append({
                "source": source,
                "chunk_index": chunk_index,
            })

    openai_client = OpenAI()
    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))

    if reset:
        try:
            chroma_client.delete_collection(collection_name)
        except ValueError:
            pass

    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"embedding_model": embedding_model},
    )

    embeddings = embed_texts(
        client=openai_client,
        texts=chunks,
        model=embedding_model,
        batch_size=batch_size,
    )

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"Indexed {len(documents)} documents")
    print(f"Stored {len(chunks)} chunks")
    print(f"Collection: {collection_name}")
    print(f"Chroma path: {chroma_dir}")
    print(f"Embedding model: {embedding_model}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index local documents into Chroma for RAG.")
    parser.add_argument("--documents-dir", default=DEFAULT_DOCUMENTS_DIR)
    parser.add_argument("--chroma-dir", default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--chunk-size", type=int, default=350)
    parser.add_argument("--overlap", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-reset", action="store_true", help="Keep existing collection data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_index(
        documents_dir=Path(args.documents_dir),
        chroma_dir=Path(args.chroma_dir),
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        reset=not args.no_reset,
    )


if __name__ == "__main__":
    main()
