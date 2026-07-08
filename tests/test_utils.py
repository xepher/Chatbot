# encoding=utf-8
import pytest
import torch
from utils import AverageMeter, unicodeToAscii, normalizeString, maskNLLLoss, find_checkpoint

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


def test_find_checkpoint(tmp_path):
    import os

    # Test directory doesn't exist
    assert find_checkpoint(os.path.join(tmp_path, "nonexistent")) is None

    # Test empty directory
    os.makedirs(os.path.join(tmp_path, "empty"))
    assert find_checkpoint(os.path.join(tmp_path, "empty")) is None

    # Create dummy checkpoints
    dir_path = os.path.join(tmp_path, "checkpoints")
    os.makedirs(dir_path)

    open(os.path.join(dir_path, "checkpoint_1_4.500.tar"), 'w').close()
    open(os.path.join(dir_path, "checkpoint_2_3.200.tar"), 'w').close()
    open(os.path.join(dir_path, "checkpoint_3_3.800.tar"), 'w').close()

    # Best (lowest val_loss) should be checkpoint_2_3.200.tar
    best_cp = find_checkpoint(dir_path, mode='best')
    assert os.path.basename(best_cp) == "checkpoint_2_3.200.tar"

    # Latest (highest epoch) should be checkpoint_3_3.800.tar
    latest_cp = find_checkpoint(dir_path, mode='latest')
    assert os.path.basename(latest_cp) == "checkpoint_3_3.800.tar"


def test_native_dataloader():
    from data_gen import ChatbotDataset, batch2TrainData
    from torch.utils.data import DataLoader

    # Ensure dataset loads correctly
    dataset = ChatbotDataset('valid')
    assert len(dataset) > 0

    # Ensure a single item is returned as a tuple of lists
    sample = dataset[0]
    assert isinstance(sample, tuple)
    assert len(sample) == 2
    assert isinstance(sample[0], list)
    assert isinstance(sample[1], list)

    # Test DataLoader with collate_fn=batch2TrainData
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=batch2TrainData)
    batch = next(iter(loader))
    
    inp, lengths, output, mask, max_target_len = batch
    assert inp.shape[1] == 4
    assert len(lengths) == 4
    assert output.shape[1] == 4
    assert mask.shape[1] == 4



