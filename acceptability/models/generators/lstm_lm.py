import torch

from torch import nn
from torch.autograd import Variable


class LSTMLanguageModel(nn.Module):
    def __init__(self, emb_dim, seq_length, hidden_dim, batch_size, vocab_size,
                 num_layers, dropout=0.5, bidirectional=False):
        super(LSTMLanguageModel, self).__init__()
        self.emb_dim = emb_dim
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=dropout)

        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=self.hidden_dim,
                            num_layers=num_layers, dropout=dropout,
                            bidirectional=bidirectional)
        self.fc = nn.Linear(self.hidden_dim, vocab_size)

        self.init_weights()

    def init_weights(self):
        init_range = 0.1

        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.fill_(0.0)

    def forward(self, x, hidden):
        x = self.dropout(self.embedding(x))
        out, hidden = self.lstm(x, hidden)
        out = self.dropout(out)
        logits = self.fc(out.view(-1, self.hidden_dim))
        return logits, hidden

    def init_hidden(self, bsz=None):
        if not bsz:
            bsz = self.batch_size
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.num_layers, bsz, self.hidden_dim).zero_()),
                Variable(weight.new(self.num_layers, bsz, self.hidden_dim).zero_()))
