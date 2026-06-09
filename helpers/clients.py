from __future__ import annotations

import os

from openai import OpenAI

from helpers.config import DEFAULT_DEEPSEEK_BASE_URL
from helpers.utils import estimate_tokens, usage_value


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
