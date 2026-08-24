"""Claude API access: structured outputs with vision support and refusal fallback."""

from __future__ import annotations

import base64
from functools import lru_cache
from typing import TypeVar

import anthropic
from pydantic import BaseModel

from . import config

T = TypeVar("T", bound=BaseModel)


class RequestRefusedError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def get_client() -> anthropic.Anthropic:
    # Credentials resolve from ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN /
    # an `ant auth login` profile — never hardcode a key.
    return anthropic.Anthropic()


def image_block(png_bytes: bytes) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.standard_b64encode(png_bytes).decode("utf-8"),
        },
    }


def run_structured(system: str, content: str | list[dict], output_format: type[T]) -> T:
    """One structured request; on a safety refusal, retry once on the fallback model."""
    client = get_client()
    last_refusal = ""
    for model in (config.LLM_MODEL, config.LLM_FALLBACK_MODEL):
        response = client.messages.parse(
            model=model,
            max_tokens=config.LLM_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": content}],
            output_format=output_format,
        )
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            last_refusal = getattr(details, "explanation", None) or "no explanation provided"
            continue
        if response.parsed_output is None:
            raise RuntimeError(
                f"Model {model} returned no parsable output (stop_reason={response.stop_reason})"
            )
        return response.parsed_output
    raise RequestRefusedError(
        f"Both {config.LLM_MODEL} and {config.LLM_FALLBACK_MODEL} refused: {last_refusal}"
    )
