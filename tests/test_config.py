import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import OCRConfig
import pytest

def test_default_config_is_valid():
    config = OCRConfig()
    config.validate()  # should not raise

def test_invalid_engine_raises():
    config = OCRConfig(engine="unknown")
    with pytest.raises(ValueError):
        config.validate()

def test_invalid_confidence_raises():
    config = OCRConfig(min_confidence=1.5)
    with pytest.raises(ValueError):
        config.validate()

def test_low_dpi_raises():
    config = OCRConfig(dpi=30)
    with pytest.raises(ValueError):
        config.validate()

def test_config_default_formats():
    config = OCRConfig()
    assert ".pdf" in config.supported_formats
    assert ".jpg" in config.supported_formats
