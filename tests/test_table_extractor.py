import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.table_extractor import TableExtractor, TableCell

def test_detect_rows_basic():
    extractor = TableExtractor(min_cols=2)
    lines = ["Item  Qty  Price", "Widget  5  €10.00", "Gadget  2  €25.00"]
    rows = extractor.detect_rows(lines)
    assert len(rows) == 3
    assert rows[0][0] == "Item"

def test_detect_rows_ignores_single_column():
    extractor = TableExtractor(min_cols=2)
    lines = ["Header Only", "Col1  Col2"]
    rows = extractor.detect_rows(lines)
    assert len(rows) == 1

def test_to_csv_string():
    extractor = TableExtractor()
    rows = [["Name", "Value"], ["foo", "bar"]]
    csv = extractor.to_csv_string(rows)
    assert "Name,Value" in csv
    assert "foo,bar" in csv

def test_table_cell_to_dict():
    cell = TableCell(0, 0, "Header")
    d = cell.to_dict()
    assert d["text"] == "Header"
    assert d["row"] == 0
