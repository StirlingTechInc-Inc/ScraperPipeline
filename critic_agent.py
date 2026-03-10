"""Critic agent for adversarial validation."""

from __future__ import annotations

import json
import logging
from typing import Any

import requests

from config import CriticResult, PipelineConfig


LOGGER = logging.getLogger(__name__)


CRITIC_SYSTEM_PROMPT = """
You are a financial fact-checking agent in an adversarial validation loop.
Compare the generated summary against the source article.
Reject the summary if any financial metric, company name, ticker, date, entity, event, or causal claim is missing, distorted, or unsupported.
Return only valid JSON with this schema:
{
  "valid": true,
  "error": ""
}
If the summary is invalid, set "valid" to false and provide a concise error trace in "error".
""".strip()


def validate_summary(
    source_text: str,
    summary: str,
    config: PipelineConfig | None = None,
) -> CriticResult:
    """Validate a generated summary against its source article."""

    runtime_config = config or PipelineConfig()
    payload = {
        "model": runtime_config.critic_model,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Source article text:\n"
                    f"{source_text}\n\n"
                    "Generated summary:\n"
                    f"{summary}\n\n"
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
        LOGGER.error("Critic agent request failed: %s", exc)
        return CriticResult(valid=False, error=f"Critic request failed: {exc}")

    try:
        content = response.json()["choices"][0]["message"]["content"]
        parsed = _extract_json_object(content)
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        LOGGER.error("Critic agent returned malformed JSON: %s", exc)
        return CriticResult(valid=False, error="Critic returned malformed JSON")

    valid = bool(parsed.get("valid", False))
    error = str(parsed.get("error", "")).strip() or None
    return CriticResult(valid=valid, error=error if not valid else None)


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
