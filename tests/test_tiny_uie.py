"""Tests for tiny-uie package."""

import tiny_uie


def test_import() -> None:
    """Test that the package can be imported and has a version."""
    version = tiny_uie.__version__
    assert isinstance(version, str)
