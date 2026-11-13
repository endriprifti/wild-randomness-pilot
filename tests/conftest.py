"""Unit tests run from the pilot folder: `python -m pytest -q` (add `-m "not slow"` to skip the slow ones)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def rng():
    return np.random.default_rng(20261105)
