from finrag.models import ElementType
from finrag.parsing import extract_figures, extract_tables, extract_text, parse_pdf


def test_extract_text_finds_narrative(sample_pdf):
    elements = extract_text(sample_pdf)
    joined = " ".join(e.text for e in elements)
    assert "4.2 million" in joined
    assert "31 percent" in joined
    assert all(e.type is ElementType.TEXT for e in elements)


def test_extract_tables_produces_markdown(sample_pdf):
    tables = extract_tables(sample_pdf)
    assert len(tables) == 1
    md = tables[0].text
    assert md.startswith("| Metric |")
    assert "| Revenue ($M) | 3.2 | 3.8 | 4.2 |" in md
    assert tables[0].page == 2


def test_extract_figures_finds_chart_image(sample_pdf):
    figures = extract_figures(sample_pdf)
    assert len(figures) >= 1
    chart = figures[0]
    assert chart.type is ElementType.CHART
    assert chart.image_png and chart.image_png[:8] == b"\x89PNG\r\n\x1a\n"
    assert chart.page == 3


def test_parse_pdf_combines_all_modalities(sample_pdf):
    elements = parse_pdf(sample_pdf)
    types = {e.type for e in elements}
    assert types == {ElementType.TEXT, ElementType.TABLE, ElementType.CHART}
    assert all(e.source == sample_pdf.name for e in elements)
