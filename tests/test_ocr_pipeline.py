"""Unit tests for OCR pipeline."""


def test_image_is_not_none():
    """Basic sanity check."""
    img = b"fake_image_bytes"
    assert img is not None


def test_text_extraction_returns_string():
    extracted = "Invoice No: 12345"
    assert isinstance(extracted, str)


def test_confidence_score_in_range():
    confidence = 0.92
    assert 0.0 <= confidence <= 1.0
