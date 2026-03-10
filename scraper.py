"""RSS and article scraping for the ingestion pipeline."""

from __future__ import annotations

import logging
from typing import List

import feedparser
import requests
from bs4 import BeautifulSoup
from requests import Response

from config import Article, PipelineConfig


LOGGER = logging.getLogger(__name__)


def fetch_articles(config: PipelineConfig | None = None) -> List[Article]:
    """Fetch and extract the top articles from the configured RSS feed."""

    runtime_config = config or PipelineConfig()
    headers = {"User-Agent": runtime_config.user_agent}
    articles: list[Article] = []

    try:
        response = requests.get(
            runtime_config.rss_feed_url,
            timeout=runtime_config.request_timeout_seconds,
            headers=headers,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        LOGGER.error("RSS feed request failed: %s", exc)
        return articles

    feed = feedparser.parse(response.content)
    if getattr(feed, "bozo", False):
        LOGGER.warning("RSS feed parsing reported malformed content: %s", getattr(feed, "bozo_exception", "unknown"))

    for entry in feed.entries[: runtime_config.max_articles]:
        article_url = getattr(entry, "link", "").strip()
        title = getattr(entry, "title", "").strip()
        if not article_url or not title:
            LOGGER.warning("Skipping malformed RSS entry with missing title or URL")
            continue

        article_text = _fetch_article_text(
            article_url,
            timeout_seconds=runtime_config.request_timeout_seconds,
            headers=headers,
        )
        if not article_text:
            LOGGER.warning("Skipping article with empty extracted body: %s", article_url)
            continue

        articles.append(Article(title=title, url=article_url, text=article_text))

    LOGGER.info("Fetched %s articles from feed", len(articles))
    return articles


def _fetch_article_text(url: str, timeout_seconds: int, headers: dict[str, str]) -> str:
    """Fetch an article page and extract its textual content."""

    try:
        response = requests.get(url, timeout=timeout_seconds, headers=headers)
        response.raise_for_status()
    except requests.Timeout:
        LOGGER.warning("Timed out fetching article: %s", url)
        return ""
    except requests.RequestException as exc:
        LOGGER.warning("Article request failed for %s: %s", url, exc)
        return ""

    return _extract_text_from_html(response)


def _extract_text_from_html(response: Response) -> str:
    """Extract article text from HTML content using structural fallbacks."""

    try:
        soup = BeautifulSoup(response.text, "html.parser")
    except Exception as exc:
        LOGGER.warning("HTML parsing failed for %s: %s", response.url, exc)
        return ""

    container = (
        soup.find("article")
        or soup.find("main")
        or soup.find(attrs={"role": "main"})
        or soup.body
    )
    if container is None:
        LOGGER.warning("No parseable container found for %s", response.url)
        return ""

    paragraphs = container.find_all("p")
    if not paragraphs:
        LOGGER.warning("No paragraph tags found for %s", response.url)
        return ""

    text = " ".join(
        paragraph.get_text(" ", strip=True)
        for paragraph in paragraphs
        if paragraph.get_text(" ", strip=True)
    )
    return " ".join(text.split())
