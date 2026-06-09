from __future__ import annotations

from helpers.cli import parse_args, query_rag
from helpers.utils import resolve_project_path


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
            retrieval_mode=args.retrieval_mode,
            candidate_k=args.candidate_k,
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
            retrieval_mode=args.retrieval_mode,
            candidate_k=args.candidate_k,
            show_stats=not args.no_stats,
        )


if __name__ == "__main__":
    main()
