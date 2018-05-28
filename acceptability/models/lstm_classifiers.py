import torch
from torch import nn
from acceptability.models import ELMOClassifier


class LSTMClassifier(nn.Module):
    def __init__(self, hidden_size, embedding_size, num_layers,
                 reduction_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.reduction_size = reduction_size
        self.ih2h = nn.LSTM(embedding_size, hidden_size,
                            num_layers=num_layers,
                            bidirectional=True, batch_first=True)
        self.h2r = nn.Linear(2 * hidden_size, reduction_size)
        self.r2o = nn.Linear(reduction_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x, hidden_states):
        o, _ = self.ih2h(x, hidden_states)
        reduction = self.sigmoid(self.h2r(o[-1]))
        output = self.sigmoid(self.r2o(reduction))
        return output, reduction


class LSTMPoolingClassifier(nn.Module):
    def __init__(self, hidden_size, embedding_size, num_layers, dropout=0.5):
        super(LSTMPoolingClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.ih2h = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.pool2o = nn.Linear(2 * hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        o, _ = self.ih2h(x)
        pool = nn.functional.max_pool1d(o.transpose(1, 2), x.shape[1])
        pool = pool.transpose(1, 2).squeeze()
        pool = self.dropout(pool)
        output = self.sigmoid(self.pool2o(pool))
        return output.squeeze(), pool


class LSTMPoolingClassifierWithELMo(nn.Module):
    def __init__(self, lm_path, hidden_size, num_layers, dropout=0.5):
        super(LSTMPoolingClassifierWithELMo, self).__init__()

        self.elmo = ELMOClassifier(lm_path, hidden_size, dropout)

        # Embedding dim would be hidden dim of the ELMoClassifier
        self.embedding_size = self.elmo.hidden_dim
        self.pooling_classifier = LSTMPoolingClassifier(hidden_size, self.embedding_size,
                                                        num_layers, dropout)

    def forward(self, x):
        _, x = self.elmo(x)

        return self.pooling_classifier(x)
