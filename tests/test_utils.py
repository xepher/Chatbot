# encoding=utf-8
import pytest
import torch
from utils import AverageMeter, unicodeToAscii, normalizeString, maskNLLLoss

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


def test_mask_nll_loss():
    # inp shape: (batch_size, vocab_size)
    inp = torch.tensor([[0.2, 0.8], [0.6, 0.4]], dtype=torch.float32)
    # target shape: (batch_size)
    target = torch.tensor([1, 0], dtype=torch.long)
    # mask shape: (batch_size) as ByteTensor (reproducing the error)
    mask = torch.tensor([1, 0], dtype=torch.uint8)
    
    # This should run without throwing RuntimeError
    loss, nTotal = maskNLLLoss(inp, target, mask)
    assert nTotal == 1
    assert isinstance(loss, torch.Tensor)

