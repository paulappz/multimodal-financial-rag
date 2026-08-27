"""Claude API access: structured outputs with vision support and refusal fallback.

Two interchangeable backends, picked from the environment:
- First-party Claude API (default): structured outputs via messages.parse.
- Amazon Bedrock (when AWS_BEARER_TOKEN_BEDROCK is set and no first-party
  credentials are): the Mantle messages endpoint. Bedrock does not accept
  output_config.format, so structured output uses a forced tool call whose
  input_schema is the same Pydantic schema, validated on return. Vision
  content blocks work identically on both backends.
"""

from __future__ import annotations

import base64
import os
from functools import lru_cache
from typing import TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from . import config

T = TypeVar("T", bound=BaseModel)


class RequestRefusedError(RuntimeError):
    pass


def _use_bedrock() -> bool:
    first_party = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    return bool(os.environ.get("AWS_BEARER_TOKEN_BEDROCK")) and not first_party


@lru_cache(maxsize=1)
def get_client() -> anthropic.Anthropic:
    if _use_bedrock():
        from anthropic import AnthropicBedrockMantle

        region = os.environ.get("BEDROCK_REGION") or os.environ.get("AWS_REGION") or "us-east-1"
        return AnthropicBedrockMantle(aws_region=region)
    # Credentials resolve from ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN /
    # an `ant auth login` profile — never hardcode a key.
    return anthropic.Anthropic()


def _qualify(model: str) -> str:
    if _use_bedrock() and not model.startswith("anthropic."):
        return f"anthropic.{model}"
    return model


def image_block(png_bytes: bytes) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.standard_b64encode(png_bytes).decode("utf-8"),
        },
    }


def _request(client, model: str, system: str, content, output_format: type[T]):
    if not _use_bedrock():
        return client.messages.parse(
            model=model,
            max_tokens=config.LLM_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": content}],
            output_format=output_format,
        )
    return client.messages.create(
        model=model,
        max_tokens=config.LLM_MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": content}],
        tools=[
            {
                "name": "emit_result",
                "description": "Report the structured result of this task.",
                "input_schema": output_format.model_json_schema(),
            }
        ],
        tool_choice={"type": "tool", "name": "emit_result"},
    )


def _extract(response, output_format: type[T], model: str) -> T:
    if not _use_bedrock():
        if response.parsed_output is None:
            raise RuntimeError(
                f"Model {model} returned no parsable output (stop_reason={response.stop_reason})"
            )
        return response.parsed_output
    block = next((b for b in response.content if b.type == "tool_use"), None)
    if block is None:
        raise RuntimeError(
            f"Model {model} returned no tool call (stop_reason={response.stop_reason})"
        )
    try:
        return output_format.model_validate(block.input)
    except ValidationError as exc:
        raise RuntimeError(f"Model {model} returned schema-invalid output: {exc}") from exc


def run_structured(system: str, content: str | list[dict], output_format: type[T]) -> T:
    """One structured request; on a safety refusal, retry once on the fallback model."""
    client = get_client()
    last_refusal = ""
    for model in (config.LLM_MODEL, config.LLM_FALLBACK_MODEL):
        response = _request(client, _qualify(model), system, content, output_format)
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            last_refusal = getattr(details, "explanation", None) or "no explanation provided"
            continue
        return _extract(response, output_format, model)
    raise RequestRefusedError(
        f"Both {config.LLM_MODEL} and {config.LLM_FALLBACK_MODEL} refused: {last_refusal}"
    )
