import json
from pathlib import Path

import pytest

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA


@pytest.fixture(scope="session")
def reference() -> dict:
    """Results of the JV Analysis Hub v1.0.0 program on the files in tests/data."""
    return json.loads((DATA / "reference_v1_0_0.json").read_text())
