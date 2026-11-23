"""Bedrock structured output arrives as a forced tool call; tolerate one extra nesting level."""

from types import SimpleNamespace

import pytest

from finrag import llm
from finrag.models import ChartDescription

GOOD = {"chart_type": "bar chart", "title": "Deliveries", "description": "Quarterly deliveries.",
        "key_values": ["Q3-2026: 497k"], "trend": "Up."}


def _tool_response(payload):
    return SimpleNamespace(content=[SimpleNamespace(type="tool_use", input=payload)], stop_reason="tool_use")


@pytest.fixture(autouse=True)
def bedrock(monkeypatch):
    monkeypatch.setattr(llm, "_use_bedrock", lambda: True)


def test_plain_tool_input_validates():
    assert llm._extract(_tool_response(GOOD), ChartDescription, "m").title == "Deliveries"


def test_result_nested_one_level_is_unwrapped():
    assert llm._extract(_tool_response({"paramName": GOOD}), ChartDescription, "m").title == "Deliveries"


def test_wrong_shape_raises_structured_output_error():
    with pytest.raises(llm.StructuredOutputError):
        llm._extract(_tool_response({"paramName": {"oops": 1}}), ChartDescription, "m")
    assert issubclass(llm.StructuredOutputError, ValueError)  # per-figure skip catches it
