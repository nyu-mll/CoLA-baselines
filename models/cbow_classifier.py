import torch
from torch import nn

class CBOWClassifier(nn.Module):
    """
    Continuous bag of words classifier.
    """
    def __init__(self, hidden_size, input_size, max_pool):
        """
        :param hidden_size:
        :param input_size:
        :param max_pool: if true then max pool over word embeddings,
                         else sum word embeddings
        """
        super(CBOWClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.max_pool = max_pool
        self.i2h = nn.Linear(self.input_size, self.hidden_size)
        self.h2o = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        if self.max_pool:
            encoding = nn.functional.max_pool1d(x.transpose(1, 2),
                                                x.shape[1])
            encoding = encoding.transpose(1, 2).squeeze()
        else:
            encoding = x.sum(1)
        hidden = self.tanh(self.i2h(encoding))
        out = self.sigmoid(self.h2o(hidden))
        return out
