# encoding=utf-8
import pytest
from utils import AverageMeter, unicodeToAscii, normalizeString

def test_average_meter():
    meter = AverageMeter()
    assert meter.val == 0
    assert meter.avg == 0
    
    meter.update(10, 1)
    assert meter.val == 10
    assert meter.avg == 10
    
    meter.update(20, 2)
    assert meter.val == 20
    assert meter.sum == 50
    assert meter.count == 3
    assert pytest.approx(meter.avg) == 50 / 3

def test_unicode_to_ascii():
    assert unicodeToAscii("München") == "Munchen"

def test_normalize_string():
    # normalizeString processes ASCII/English normalization in this codebase
    assert normalizeString("Hello, World!") == "hello world !"
