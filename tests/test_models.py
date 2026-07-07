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
