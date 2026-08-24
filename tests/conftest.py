import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from make_sample_report import make_report  # noqa: E402


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("data") / "sample_report.pdf"
    make_report(path)
    return path
