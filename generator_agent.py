"""Generator agent for financial briefing synthesis."""

from __future__ import annotations

import json
import logging
from typing import Any

import requests

from config import PipelineConfig, SummaryMetadata, SummaryResult


LOGGER = logging.getLogger(__name__)


GENERATOR_SYSTEM_PROMPT = """
You are a financial news synthesis agent for a production ingestion pipeline.
Summarize the source article into a factual, conversational audio briefing script that can be spoken in roughly 30 seconds.
Do not speculate or introduce facts not supported by the source.
Return only valid JSON with this schema:
{
  "summary": "string",
  "metadata": {
    "company": "string",
    "sector": "string",
    "topic": "string",
    "company_ticker": "string"
  }
}
If a field is unknown, use an empty string.
""".strip()


def generate_summary(article_text: str, config: PipelineConfig | None = None) -> SummaryResult:
    """Generate a conversational summary and routing metadata from source text."""

    runtime_config = config or PipelineConfig()
    payload = {
        "model": runtime_config.groq_model,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Source article text:\n"
                    f"{article_text}\n\n"
                    "Return the JSON object only."
                ),
            },
        ],
    }

    try:
        response = requests.post(
            f"{runtime_config.groq_base_url}/chat/completions",
            headers=_groq_headers(runtime_config),
            json=payload,
            timeout=runtime_config.request_timeout_seconds,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        LOGGER.error("Generator agent request failed: %s", exc)
        raise RuntimeError("Generator agent request failed") from exc

    try:
        content = response.json()["choices"][0]["message"]["content"]
        parsed = _extract_json_object(content)
        metadata = parsed.get("metadata", {})
        return SummaryResult(
            summary=str(parsed.get("summary", "")).strip(),
            metadata=SummaryMetadata(
                company=str(metadata.get("company", "")).strip(),
                sector=str(metadata.get("sector", "")).strip(),
                topic=str(metadata.get("topic", "")).strip(),
                company_ticker=str(metadata.get("company_ticker", "")).strip(),
            ),
        )
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        LOGGER.error("Generator agent returned malformed JSON: %s", exc)
        raise RuntimeError("Generator agent returned malformed JSON") from exc


def _groq_headers(config: PipelineConfig) -> dict[str, str]:
    """Build request headers for Groq API calls."""

    if not config.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not configured")
    return {
        "Authorization": f"Bearer {config.groq_api_key}",
        "Content-Type": "application/json",
    }


def _extract_json_object(content: str) -> dict[str, Any]:
    """Parse the first JSON object from a model response."""

    stripped = content.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or start >= end:
            raise
        return json.loads(stripped[start : end + 1])
