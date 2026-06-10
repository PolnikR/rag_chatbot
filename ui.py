from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from helpers.config import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_COLLECTION,
    DEFAULT_DEEPSEEK_LLM_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_OPENAI_LLM_MODEL,
)
from helpers.rag_runner import run_rag
from helpers.utils import resolve_project_path


def default_llm_model(provider: str) -> str:
    if provider == "deepseek":
        return os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_LLM_MODEL)
    return os.getenv("OPENAI_LLM_MODEL", DEFAULT_OPENAI_LLM_MODEL)


def render_sources(retrieval_results) -> None:
    for index, result in enumerate(retrieval_results, start=1):
        source = result.metadata.get("source", "unknown")
        chunk_index = result.metadata.get("chunk_index", "?")
        mode = result.metadata.get("retrieval_mode", "retrieval")
        reranker = result.metadata.get("reranker")
        pre_rerank = result.metadata.get("pre_rerank_score")
        score_label = f"{mode} score {result.score:.4f}"
        if reranker:
            score_label += f" | reranker {reranker}"
        if isinstance(pre_rerank, float):
            score_label += f" | pre-rerank {pre_rerank:.4f}"

        with st.expander(f"Source {index}: {source} / chunk {chunk_index} / {score_label}"):
            st.write(result.text)


def main() -> None:
    load_dotenv()

    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="",
        layout="centered",
    )

    st.markdown(
        """
        <style>
        .block-container {
            max-width: 920px;
            padding-top: 2rem;
            padding-bottom: 7rem;
        }
        .stChatMessage {
            background: transparent;
        }
        div[data-testid="stChatInput"] {
            max-width: 820px;
            margin: 0 auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.title("RAG Chatbot")

    with st.sidebar:
        st.subheader("Retrieval")
        retrieval_mode = st.selectbox(
            "Mode",
            options=["hybrid", "vector", "bm25"],
            index=0,
        )
        top_k = st.slider("Top K", min_value=1, max_value=15, value=5)
        candidate_k = st.slider("Candidate K", min_value=5, max_value=50, value=20)
        reranker_mode = st.selectbox(
            "Reranker",
            options=[
                "simple",
                "none",
                "cross_encoder",
                "cohere",
                "voyage",
                "jina",
                "bge",
                "colbert",
            ],
            index=0,
            disabled=retrieval_mode != "hybrid",
        )

        st.subheader("Models")
        llm_provider = st.selectbox(
            "LLM provider",
            options=["deepseek", "openai"],
            index=0 if DEFAULT_LLM_PROVIDER == "deepseek" else 1,
        )
        llm_model = st.text_input("LLM model", value=default_llm_model(llm_provider))
        embedding_model = st.text_input("Embedding model", value=DEFAULT_EMBEDDING_MODEL)

        st.subheader("Chroma")
        chroma_dir = st.text_input("Chroma dir", value=DEFAULT_CHROMA_DIR)
        collection_name = st.text_input("Collection", value=DEFAULT_COLLECTION)

        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("retrieval_results"):
                render_sources(message["retrieval_results"])

    prompt = st.chat_input("Spytaj sa hocico...")
    if not prompt:
        return

    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"Searching with {retrieval_mode}..."):
            try:
                result = run_rag(
                    question=prompt,
                    chroma_dir=resolve_project_path(chroma_dir),
                    collection_name=collection_name,
                    embedding_model=embedding_model,
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    top_k=top_k,
                    retrieval_mode=retrieval_mode,
                    candidate_k=candidate_k,
                    reranker_mode=reranker_mode,
                )
            except Exception as exc:
                st.error(str(exc))
                return

        st.markdown(result["answer"])
        render_sources(result["retrieval_results"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "retrieval_results": result["retrieval_results"],
        "stats": result["stats"],
    })


if __name__ == "__main__":
    main()
