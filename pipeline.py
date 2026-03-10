"""Pipeline orchestration for EuphorAI ingestion."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from config import PipelineConfig, SummaryResult, configure_logging
from critic_agent import validate_summary
from generator_agent import generate_summary
from scraper import fetch_articles
from vector_store import store_vector


LOGGER = logging.getLogger(__name__)


def run_pipeline(config: PipelineConfig | None = None) -> list[dict[str, Any]]:
    """Execute the end-to-end ingestion pipeline."""

    configure_logging()
    runtime_config = config or PipelineConfig()
    articles = fetch_articles(runtime_config)
    stored_records: list[dict[str, Any]] = []

    for article in articles:
        LOGGER.info("Processing article: %s", article.title)
        summary_result = _generate_with_retry(article.text, runtime_config)
        if summary_result is None:
            LOGGER.warning("Generation failed after retries for article: %s", article.url)
            continue

        metadata = _build_storage_metadata(article.title, article.url, summary_result)
        if store_vector(summary_result, metadata, runtime_config):
            stored_records.append(
                {
                    "article": article.to_dict(),
                    "summary_result": summary_result.to_dict(),
                    "metadata": metadata,
                }
            )

    LOGGER.info("Pipeline completed with %s stored records", len(stored_records))
    return stored_records


def _generate_with_retry(
    article_text: str,
    config: PipelineConfig,
) -> SummaryResult | None:
    """Generate and validate a summary with bounded retries."""

    for attempt in range(1, config.max_generation_attempts + 1):
        try:
            summary_result = generate_summary(article_text, config)
        except RuntimeError as exc:
            LOGGER.warning("Generation attempt %s failed: %s", attempt, exc)
            continue

        critic_result = validate_summary(article_text, summary_result.summary, config)
        if critic_result.valid:
            LOGGER.info("Summary validation passed on attempt %s", attempt)
            return summary_result

        LOGGER.warning(
            "Summary validation failed on attempt %s: %s",
            attempt,
            critic_result.error,
        )

    return None


def _build_storage_metadata(
    title: str,
    url: str,
    summary_result: SummaryResult,
) -> dict[str, Any]:
    """Build metadata for vector storage and later filtering."""

    timestamp = datetime.now(timezone.utc).isoformat()
    metadata = summary_result.metadata.to_dict()
    metadata["timestamp"] = timestamp
    metadata["title"] = title
    metadata["url"] = url
    return metadata


if __name__ == "__main__":
    run_pipeline()
