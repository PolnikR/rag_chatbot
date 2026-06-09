from __future__ import annotations

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
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
