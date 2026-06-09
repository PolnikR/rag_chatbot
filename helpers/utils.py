from __future__ import annotations

from pathlib import Path

from helpers.config import PROJECT_DIR


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
