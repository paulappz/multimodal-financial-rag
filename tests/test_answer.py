from unittest.mock import patch

from finrag.answer import answer_question, build_context_block
from finrag.models import RAGAnswer, RetrievedChunk


def _chunk(text, page=1, element_type="text"):
    return RetrievedChunk(text=text, score=0.9, source="r.pdf", page=page, element_type=element_type)


def test_context_blocks_are_numbered_with_provenance():
    ctx = build_context_block([_chunk("Revenue was $4.2M."), _chunk("| a | b |", page=2, element_type="table")])
    assert "[1] (text, r.pdf p.1)" in ctx
    assert "[2] (table, r.pdf p.2)" in ctx


def test_answer_maps_citations_back_to_chunks():
    chunks = [_chunk("Revenue was $4.2M."), _chunk("Margin was 66%.", page=2)]
    fake = RAGAnswer(answer="Revenue was $4.2M [1].", cited_blocks=[1], insufficient_context=False)

    with patch("finrag.answer.run_structured", return_value=fake) as mock_run:
        result, cited = answer_question("What was revenue?", chunks)

    assert result.answer.startswith("Revenue")
    assert len(cited) == 1 and cited[0].page == 1
    prompt = mock_run.call_args.args[1]
    assert "Question: What was revenue?" in prompt
    assert "[2] (text, r.pdf p.2)" in prompt


def test_out_of_range_citations_ignored():
    chunks = [_chunk("x")]
    fake = RAGAnswer(answer="…", cited_blocks=[1, 7], insufficient_context=False)
    with patch("finrag.answer.run_structured", return_value=fake):
        _, cited = answer_question("q", chunks)
    assert len(cited) == 1
