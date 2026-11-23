from unittest.mock import patch

from finrag.charts import describe_all
from finrag.models import ChartDescription, ElementType, ParsedElement


def test_describe_all_fills_chart_text_only():
    chart = ParsedElement(type=ElementType.CHART, page=3, source="r.pdf", image_png=b"\x89PNGfake")
    text = ParsedElement(type=ElementType.TEXT, page=1, source="r.pdf", text="hello")
    desc = ChartDescription(
        chart_type="line chart",
        title="Revenue Trend",
        description="Revenue rises from $3.4M to $4.2M across four quarters.",
        key_values=["Q2'26: $4.2M"],
        trend="Steady quarter-over-quarter growth.",
    )

    with patch("finrag.charts.describe_chart", return_value=desc) as mock_describe:
        result = describe_all([chart, text])

    mock_describe.assert_called_once_with(b"\x89PNGfake")
    assert "[line chart] Revenue Trend" in chart.text
    assert "Key values: Q2'26: $4.2M" in chart.text
    assert text.text == "hello"
    assert result[0] is chart


def test_photos_are_left_out_of_the_index():
    photo = ParsedElement(type=ElementType.CHART, page=11, source="r.pdf", image_png=b"\x89PNGfake")
    desc = ChartDescription(chart_type="Other", title="Vehicle unveiling at night",
                            description="A promotional photo; not a data visualization.",
                            key_values=[], trend="None.")
    with patch("finrag.charts.describe_chart", return_value=desc):
        describe_all([photo])
    assert photo.text == ""  # no text, so chunking skips it
