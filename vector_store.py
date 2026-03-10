"""Pinecone integration for verified briefing storage."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any
from uuid import uuid4

from config import PipelineConfig, SummaryResult


LOGGER = logging.getLogger(__name__)


def store_vector(
    summary: SummaryResult,
    metadata: dict[str, Any],
    config: PipelineConfig | None = None,
) -> bool:
    """Embed verified text and upsert it into Pinecone."""

    runtime_config = config or PipelineConfig()

    try:
        embedding = _embed_text(summary.summary, runtime_config)
    except Exception as exc:
        LOGGER.error("Embedding generation failed: %s", exc)
        return False

    try:
        from pinecone import Pinecone
    except ImportError as exc:
        LOGGER.error("Pinecone client is not installed: %s", exc)
        return False

    if not runtime_config.pinecone_api_key:
        LOGGER.error("PINECONE_API_KEY is not configured")
        return False

    payload_metadata = dict(metadata)
    payload_metadata["summary"] = summary.summary

    vector = {
        "id": str(uuid4()),
        "values": embedding,
        "metadata": payload_metadata,
    }

    try:
        client = Pinecone(api_key=runtime_config.pinecone_api_key)
        index = client.Index(runtime_config.pinecone_index_name)
        index.upsert(vectors=[vector], namespace=runtime_config.pinecone_namespace)
    except Exception as exc:
        LOGGER.error("Pinecone upsert failed: %s", exc)
        return False

    LOGGER.info("Stored verified summary in Pinecone")
    return True


def _embed_text(text: str, config: PipelineConfig) -> list[float]:
    """Create a dense embedding for the verified summary."""

    model = _get_embedding_model(config.embedding_model_name)
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


@lru_cache(maxsize=2)
def _get_embedding_model(model_name: str):
    """Load and cache the embedding model instance."""

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is not installed") from exc

    return SentenceTransformer(model_name)
