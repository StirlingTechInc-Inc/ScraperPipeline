"""Configuration and shared data structures for the ingestion pipeline."""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass(slots=True)
class Article:
    """Normalized scraped article payload."""

    title: str
    url: str
    text: str

    def to_dict(self) -> dict[str, str]:
        """Return a dictionary representation of the article."""

        return asdict(self)


@dataclass(slots=True)
class SummaryMetadata:
    """Metadata extracted from the generator stage."""

    company: str
    sector: str
    topic: str
    company_ticker: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, str]:
        """Return a dictionary representation of the metadata."""

        return asdict(self)


@dataclass(slots=True)
class SummaryResult:
    """Structured generator output."""

    summary: str
    metadata: SummaryMetadata

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the summary result."""

        return {
            "summary": self.summary,
            "metadata": self.metadata.to_dict(),
        }


@dataclass(slots=True)
class CriticResult:
    """Structured critic verdict."""

    valid: bool
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the critic result."""

        return asdict(self)


@dataclass(slots=True)
class PipelineConfig:
    """Environment-backed runtime configuration."""

    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    pinecone_api_key: str = field(default_factory=lambda: os.getenv("PINECONE_API_KEY", ""))
    rss_feed_url: str = field(
        default_factory=lambda: os.getenv(
            "RSS_FEED_URL",
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EDJI,^GSPC,^IXIC&region=US&lang=en-US",
        )
    )
    request_timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT_SECONDS", "10"))
    )
    max_articles: int = field(default_factory=lambda: int(os.getenv("MAX_ARTICLES", "3")))
    groq_model: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "llama3-8b-8192")
    )
    critic_model: str = field(
        default_factory=lambda: os.getenv("CRITIC_MODEL", "llama3-8b-8192")
    )
    groq_base_url: str = field(
        default_factory=lambda: os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    )
    pinecone_index_name: str = field(
        default_factory=lambda: os.getenv("PINECONE_INDEX_NAME", "euphorai-briefings")
    )
    embedding_model_name: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    )
    pinecone_namespace: str = field(
        default_factory=lambda: os.getenv("PINECONE_NAMESPACE", "financial-briefings")
    )
    max_generation_attempts: int = field(
        default_factory=lambda: int(os.getenv("MAX_GENERATION_ATTEMPTS", "3"))
    )
    user_agent: str = field(
        default_factory=lambda: os.getenv(
            "HTTP_USER_AGENT",
            "EuphorAI-IngestionPipeline/1.0 (+https://stirlingtech.example)",
        )
    )

    def validate_required_keys(self) -> None:
        """Raise if mandatory credentials are not configured."""

        missing = []
        if not self.groq_api_key:
            missing.append("GROQ_API_KEY")
        if not self.pinecone_api_key:
            missing.append("PINECONE_API_KEY")
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")


def configure_logging(level: int = logging.INFO) -> None:
    """Configure process-wide logging if no handlers have been defined."""

    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
