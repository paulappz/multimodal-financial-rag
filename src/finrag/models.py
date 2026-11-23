"""Data models shared across the pipeline.

The structured-output models (ChartDescription, RAGAnswer) double as the
schemas Claude's structured outputs are validated against.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ElementType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    CHART = "chart"


class ParsedElement(BaseModel):
    """One unit extracted from a PDF: a text block, a Markdown table, or a figure."""

    type: ElementType
    page: int
    source: str
    text: str = ""
    image_png: bytes | None = Field(default=None, repr=False)


class ChartDescription(BaseModel):
    """Structured output for Claude's vision pass over a chart/figure image."""

    chart_type: str = Field(description="e.g. line chart, bar chart, pie chart, table image, other")
    title: str = Field(description="Chart title, or a short inferred one")
    description: str = Field(
        description="Faithful textual rendering of what the chart shows: axes, units, series, and the story the data tells"
    )
    key_values: list[str] = Field(
        description="Concrete data points readable from the chart, e.g. 'Q2 2026 revenue: $4.2M'"
    )
    trend: str = Field(description="One-sentence trend summary")

    @property
    def is_data(self) -> bool:
        """False for photos, logos and decoration (chart_type "other"): nothing to retrieve."""
        return self.chart_type.strip().lower() != "other"

    def as_text(self) -> str:
        lines = [f"[{self.chart_type}] {self.title}", self.description]
        if self.key_values:
            lines.append("Key values: " + "; ".join(self.key_values))
        lines.append(f"Trend: {self.trend}")
        return "\n".join(lines)


class RAGAnswer(BaseModel):
    """Structured output for answer synthesis."""

    answer: str = Field(
        description="The answer, grounded strictly in the provided context blocks; cite blocks inline as [n]"
    )
    cited_blocks: list[int] = Field(
        description="Indices of the context blocks actually used to answer"
    )
    insufficient_context: bool = Field(
        description="True when the context does not contain enough information to answer"
    )


class RetrievedChunk(BaseModel):
    text: str
    score: float
    source: str
    page: int
    element_type: str
