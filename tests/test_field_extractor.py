import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.field_extractor import extract_invoice_number, extract_date, extract_currency_amount

def test_extract_invoice_number():
    assert extract_invoice_number("Invoice No: INV-2025-001") == "INV-2025-001"
    assert extract_invoice_number("INVOICE NUMBER: 42") == "42"
    assert extract_invoice_number("no invoice here") is None

def test_extract_date_iso():
    assert extract_date("Date: 2025-11-23") == "2025-11-23"

def test_extract_date_us_format():
    assert extract_date("Date: 11/23/2025") == "11/23/2025"

def test_extract_currency_euro():
    result = extract_currency_amount("Total: €4,250.00")
    assert result is not None
    assert "4,250" in result

def test_extract_currency_usd():
    result = extract_currency_amount("Amount: $1,500.99")
    assert result is not None

def test_returns_none_when_no_match():
    assert extract_date("no date here") is None
    assert extract_currency_amount("no money here") is None
