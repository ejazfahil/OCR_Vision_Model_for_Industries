import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.metrics import character_error_rate, word_error_rate

def test_cer_identical_strings():
    assert character_error_rate("hello", "hello") == 0.0

def test_cer_completely_different():
    cer = character_error_rate("abc", "xyz")
    assert cer == 1.0

def test_cer_one_substitution():
    cer = character_error_rate("cat", "bat")
    assert abs(cer - 1/3) < 0.001

def test_wer_identical():
    assert word_error_rate("hello world", "hello world") == 0.0

def test_wer_one_wrong_word():
    wer = word_error_rate("the cat sat", "the dog sat")
    assert abs(wer - 1/3) < 0.001

def test_cer_empty_reference():
    assert character_error_rate("", "") == 0.0
