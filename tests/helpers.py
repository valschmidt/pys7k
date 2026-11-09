"""Shared test helpers: locating sample data and comparing records."""

from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLE_FILES = sorted(DATA_DIR.glob("*.s7k")) if DATA_DIR.is_dir() else []
requires_sample_data = pytest.mark.skipif(
    not SAMPLE_FILES, reason="no sample .s7k files in data/ (run scripts/make_test_subsets.py --defaults)")


def sample(name):
    """Path to a sample file in data/, skipping the test if it is missing."""
    path = DATA_DIR / name
    if not path.exists():
        pytest.skip("sample file %s not found" % name)
    return path


def assert_same(a, b, path="record"):
    """Assert two decoded structures are equal, comparing arrays by value."""
    if isinstance(a, dict):
        assert isinstance(b, dict), "%s: dict vs %r" % (path, type(b))
        assert set(a) == set(b), "%s: keys differ: %s vs %s" % (path, sorted(a), sorted(b))
        for k in a:
            assert_same(a[k], b[k], "%s.%s" % (path, k))
    elif isinstance(a, list):
        assert len(a) == len(b), "%s: lengths differ" % path
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, "%s[%d]" % (path, i))
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape, "%s: shape %s vs %s" % (path, a.shape, b.shape)
        assert np.array_equal(a, b, equal_nan=a.dtype.kind == "f"), "%s: arrays differ" % path
    elif isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        pass
    else:
        assert a == b, "%s: %r != %r" % (path, a, b)
