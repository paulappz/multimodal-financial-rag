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


def _unruled_statement(path):
    """A statement laid out like real filings: no grid lines, columns aligned only by position."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 740, " ".join("BALANCE SHEET"))  # letter-spaced heading
    c.setFont("Helvetica", 9)
    cols = [300, 360, 420, 480]
    rows = [
        ("In millions of USD", ["30-Sep-25", "31-Dec-25", "31-Mar-26", "30-Jun-26"]),
        ("Adjustments to reconcile net income to net cash provided by operating activities, in detail:", []),
        ("Cash and equivalents", ["41,647", "44,059", "44,743", "43,524"]),
        ("Inventory (1)", ["12,276", "12,392", "14,434", "13,752"]),
        ("Restructuring and other", ["238", "162", "—", "(5)"]),
        ("Total current assets", ["64,653", "68,642", "69,748", "68,758"]),
    ]
    y = 710
    for label, values in rows:
        c.drawString(50, y, label)
        for x, v in zip(cols, values):
            c.drawRightString(x + 40, y, v)
        y -= 16
    c.drawString(300, 60, "28")  # page number footer
    c.save()


def test_unruled_statement_rebuilt_from_text_rows(tmp_path):
    path = tmp_path / "statement.pdf"
    _unruled_statement(path)
    tables = extract_tables(path)
    assert len(tables) == 1
    md = tables[0].text
    assert md.splitlines()[0] == "| In millions of USD | 30-Sep-25 | 31-Dec-25 | 31-Mar-26 | 30-Jun-26 |"
    assert "| Inventory (1) | 12,276 | 12,392 | 14,434 | 13,752 |" in md  # footnote stays in the label
    assert "| Restructuring and other | 238 | 162 | — | (5) |" in md
    assert "| Total current assets | 64,653 | 68,642 | 69,748 | 68,758 |" in md
    assert "28" not in md.splitlines()[-1]
