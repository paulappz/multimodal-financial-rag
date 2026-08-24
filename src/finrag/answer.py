"""Answer synthesis: retrieved context → grounded, cited answer."""

from __future__ import annotations

from .llm import run_structured
from .models import RAGAnswer, RetrievedChunk

SYSTEM = """You answer questions about financial reports using ONLY the numbered context
blocks provided. Rules:
- Ground every claim in the context; cite blocks inline as [n] after the claim they support.
- Quote figures exactly as they appear (units, currency, periods). Never extrapolate numbers.
- Financial data is period-sensitive: if the question's period and the context's period
  differ, say so explicitly.
- If the context is insufficient or contradictory, set insufficient_context=true and say
  what is missing instead of guessing."""


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(
            f"[{i}] ({c.element_type}, {c.source} p.{c.page})\n{c.text}"
        )
    return "\n\n".join(parts)


def answer_question(question: str, chunks: list[RetrievedChunk]) -> tuple[RAGAnswer, list[RetrievedChunk]]:
    """Returns the structured answer plus the subset of chunks it actually cited."""
    prompt = (
        "Context blocks:\n\n"
        + build_context_block(chunks)
        + f"\n\nQuestion: {question}"
    )
    result = run_structured(SYSTEM, prompt, RAGAnswer)
    cited = [chunks[i - 1] for i in result.cited_blocks if 1 <= i <= len(chunks)]
    return result, cited
