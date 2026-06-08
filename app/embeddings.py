"""OpenAI embeddings via the LiteLLM proxy + cosine similarity utilities."""
from __future__ import annotations

import json
import math
import os
from typing import List

from openai import OpenAI

_client: OpenAI | None = None

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
MAX_INPUT_CHARS = 32000  # ~8K tokens


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=os.getenv("PROXY_BASE_URL", "https://proxy.npedwards.com/v1"),
            api_key=os.getenv("PROXY_API_KEY"),
        )
    return _client


def embed(text: str) -> List[float]:
    resp = _get_client().embeddings.create(
        model=EMBEDDING_MODEL,
        input=text[:MAX_INPUT_CHARS],
    )
    return resp.data[0].embedding


def embed_batch(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    resp = _get_client().embeddings.create(
        model=EMBEDDING_MODEL,
        input=[t[:MAX_INPUT_CHARS] for t in texts],
    )
    return [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def to_json(embedding: List[float]) -> str:
    return json.dumps(embedding)


def from_json(s: str | None) -> List[float] | None:
    if not s:
        return None
    return json.loads(s)
