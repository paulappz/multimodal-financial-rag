from finrag.chunking import build_nodes
from finrag.models import ElementType, ParsedElement


def _element(type_, text, page=1):
    return ParsedElement(type=type_, page=page, source="r.pdf", text=text)


def test_tables_kept_whole():
    md = "| A | B |\n|---|---|\n| 1 | 2 |"
    nodes = build_nodes([_element(ElementType.TABLE, md, page=2)])
    assert len(nodes) == 1
    assert md in nodes[0].get_content()
    assert nodes[0].metadata["element_type"] == "table"
    assert nodes[0].metadata["page"] == 2


def test_narrative_split_with_page_metadata():
    long_text = "Revenue grew strongly. " * 200
    nodes = build_nodes([_element(ElementType.TEXT, long_text, page=1)])
    assert len(nodes) > 1
    assert all(n.metadata["page"] == 1 for n in nodes)
    assert all(n.metadata["element_type"] == "text" for n in nodes)


def test_chart_descriptions_become_nodes():
    nodes = build_nodes(
        [_element(ElementType.CHART, "[line chart] Revenue Trend\nRevenue rose to $4.2M.", page=3)]
    )
    assert len(nodes) == 1
    assert "Figure (page 3):" in nodes[0].get_content()


def test_chart_without_description_skipped():
    el = ParsedElement(type=ElementType.CHART, page=3, source="r.pdf", image_png=b"png")
    assert build_nodes([el]) == []
