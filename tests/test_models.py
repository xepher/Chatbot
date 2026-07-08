# encoding=utf-8
import torch
from torch import nn
from models import EncoderRNN, LuongAttnDecoderRNN

def test_encoder_decoder_forward():
    vocab_size = 100
    hidden_size = 32
    batch_size = 4
    seq_len = 5

    embedding = nn.Embedding(vocab_size, hidden_size)

    # Instantiate models
    encoder = EncoderRNN(hidden_size, embedding, n_layers=2, dropout=0.1)
    decoder = LuongAttnDecoderRNN('dot', embedding, hidden_size, vocab_size, n_layers=2, dropout=0.1)

    # Create dummy input data (seq_len, batch_size)
    # PyTorch pack_padded_sequence requires input_lengths to be on CPU and 1D tensor
    # If enforce_sorted is True (default), they must be sorted in decreasing order
    input_seq = torch.randint(0, vocab_size, (seq_len, batch_size), dtype=torch.long)
    input_lengths = torch.tensor([5, 4, 3, 2], dtype=torch.long)

    # Forward encoder
    encoder_outputs, encoder_hidden = encoder(input_seq, input_lengths)

    # Outputs shape: (seq_len, batch_size, hidden_size)
    assert encoder_outputs.shape == (seq_len, batch_size, hidden_size)
    # Hidden shape: (n_layers * num_directions, batch_size, hidden_size)
    assert encoder_hidden.shape == (2 * 2, batch_size, hidden_size)

    # Forward decoder step by step (one token at a time)
    # decoder_input shape: (1, batch_size)
    decoder_input = torch.randint(0, vocab_size, (1, batch_size), dtype=torch.long)
    # decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Forward decoder
    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)

    # Decoder output shape: (batch_size, vocab_size)
    assert decoder_output.shape == (batch_size, vocab_size)
    # Decoder hidden shape: (n_layers, batch_size, hidden_size)
    assert decoder_hidden.shape == (2, batch_size, hidden_size)


def test_encoder_forward_cuda_lengths():
    if not torch.cuda.is_available():
        return
    vocab_size = 100
    hidden_size = 32
    batch_size = 4
    seq_len = 5

    embedding = nn.Embedding(vocab_size, hidden_size)
    encoder = EncoderRNN(hidden_size, embedding, n_layers=2, dropout=0.1)

    # Move model to CUDA
    device = torch.device("cuda")
    encoder = encoder.to(device)

    # Create dummy input data on CUDA
    input_seq = torch.randint(0, vocab_size, (seq_len, batch_size), dtype=torch.long).to(device)
    input_lengths = torch.tensor([5, 4, 3, 2], dtype=torch.long).to(device)

    # Forward encoder
    encoder_outputs, encoder_hidden = encoder(input_seq, input_lengths)

    # Check device and shape
    assert encoder_outputs.is_cuda
    assert encoder_outputs.shape == (seq_len, batch_size, hidden_size)


def test_save_load_checkpoint(tmp_path):
    import os
    from torch import optim

    vocab_size = 10
    hidden_size = 8
    embedding = nn.Embedding(vocab_size, hidden_size)

    encoder = EncoderRNN(hidden_size, embedding, n_layers=1, dropout=0.1)
    decoder = LuongAttnDecoderRNN('dot', embedding, hidden_size, vocab_size, n_layers=1, dropout=0.1)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001)

    # Save checkpoint
    checkpoint_path = os.path.join(tmp_path, "checkpoint.tar")
    torch.save({
        'epoch': 5,
        'en': encoder.state_dict(),
        'de': decoder.state_dict(),
        'en_opt': encoder_optimizer.state_dict(),
        'de_opt': decoder_optimizer.state_dict(),
        'loss': 0.5,
    }, checkpoint_path)

    assert os.path.exists(checkpoint_path)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    assert checkpoint['epoch'] == 5
    assert checkpoint['loss'] == 0.5

    # Check state loading
    encoder_new = EncoderRNN(hidden_size, embedding, n_layers=1, dropout=0.1)
    encoder_new.load_state_dict(checkpoint['en'])
    # Assert values match
    for p1, p2 in zip(encoder.parameters(), encoder_new.parameters()):
        assert torch.allclose(p1, p2)


def test_train_keyboard_interrupt(monkeypatch, tmp_path):
    import train
    import data_gen

    # Mock ChatbotDataset to be a dummy dataset that returns 1 batch
    class DummyDataset:
        def __init__(self, split):
            pass
        def __len__(self):
            return 1
        def __getitem__(self, index):
            # Return dummy values that fit the unpacking
            # input_variable, lengths, target_variable, mask, max_target_len
            import torch
            return (torch.zeros(5, 4, dtype=torch.long), torch.tensor([5, 4, 3, 2]), 
                    torch.zeros(5, 4, dtype=torch.long), torch.zeros(5, 4), 5)

    monkeypatch.setattr(data_gen, "ChatbotDataset", DummyDataset)

    # Patch the train function to raise KeyboardInterrupt
    def mock_train(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(train, "train", mock_train)
    # Patch save_dir to point to tmp_path
    monkeypatch.setattr(train, "save_dir", str(tmp_path))

    # Mock other configurations/globals
    monkeypatch.setattr(train, "start_epoch", 0)
    monkeypatch.setattr(train, "epochs", 5)
    monkeypatch.setattr(train, "loadFilename", None)

    # Call train.main() which should exit gracefully and write a checkpoint
    train.main()

    # Verify that checkpoint_interrupted.tar was created in tmp_path
    import os
    expected_path = os.path.join(tmp_path, "checkpoint_interrupted.tar")
    assert os.path.exists(expected_path)



