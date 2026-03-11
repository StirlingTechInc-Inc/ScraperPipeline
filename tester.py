"""Simple smoke test harness for the Phase 2 ingestion pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from unittest.mock import patch

from config import Article, CriticResult, PipelineConfig, SummaryMetadata, SummaryResult, configure_logging


LOGGER = logging.getLogger(__name__)


def run_mocked_pipeline_test() -> list[dict[str, object]]:
    """Execute a deterministic end-to-end smoke test with mocked downstream services."""

    from pipeline import run_pipeline

    config = PipelineConfig(
        groq_api_key="test-groq-key",
        pinecone_api_key="test-pinecone-key",
        max_articles=1,
        max_generation_attempts=3,
    )
    article = Article(
        title="Apple reports higher iPhone revenue in latest quarter",
        url="https://example.com/apple-quarterly-results",
        text=(
            "Apple reported quarterly revenue of $119.6 billion, up 2 percent year over year, "
            "as iPhone revenue reached $69.7 billion. Services revenue also rose during the quarter."
        ),
    )
    summary = SummaryResult(
        summary=(
            "Apple said quarterly revenue rose to $119.6 billion, helped by $69.7 billion in iPhone sales "
            "and continued growth in services."
        ),
        metadata=SummaryMetadata(
            company="Apple",
            sector="Technology",
            topic="Earnings",
            company_ticker="AAPL",
        ),
    )

    with (
        patch("pipeline.fetch_articles", return_value=[article]),
        patch("pipeline.generate_summary", return_value=summary),
        patch("pipeline.validate_summary", return_value=CriticResult(valid=True, error=None)),
        patch("pipeline.store_vector", return_value=True),
    ):
        return run_pipeline(config)


def run_live_scraper_probe() -> list[dict[str, str]]:
    """Run a best-effort live scraper check against the configured RSS feed."""

    from scraper import fetch_articles

    config = PipelineConfig(max_articles=3)
    articles = fetch_articles(config)
    return [
        {
            "title": article.title,
            "url": article.url,
            "text_preview": article.text[:200],
        }
        for article in articles
    ]


def main() -> int:
    """Run the requested test mode and emit a compact result payload."""

    parser = argparse.ArgumentParser(description="Phase 2 smoke test harness")
    parser.add_argument(
        "--live-scraper",
        action="store_true",
        help="Probe the configured RSS feed and article extraction live",
    )
    args = parser.parse_args()

    configure_logging()

    if args.live_scraper:
        LOGGER.info("Running live scraper probe")
        try:
            result = {
                "mode": "live_scraper_probe",
                "articles": run_live_scraper_probe(),
            }
        except ModuleNotFoundError as exc:
            result = {
                "mode": "live_scraper_probe",
                "error": f"Missing dependency: {exc.name}",
            }
    else:
        LOGGER.info("Running mocked pipeline smoke test")
        try:
            result = {
                "mode": "mocked_pipeline_smoke_test",
                "stored_records": run_mocked_pipeline_test(),
            }
        except ModuleNotFoundError as exc:
            result = {
                "mode": "mocked_pipeline_smoke_test",
                "error": f"Missing dependency: {exc.name}",
            }

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
