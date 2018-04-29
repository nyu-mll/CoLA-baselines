import torch

from torch import nn
from .lstm_classifiers import LSTMPoolingClassifier
from .generators.lstm_lm import LSTMLanguageModel

class LinearClassifier(nn.Module):
    """
    A basic linear classifier for acceptability judgments.
    Input sentence embedding (with size = 2*encoding_size,
    factor of 2 comes from bidirectional LSTM encoding)
    Hidden layer (encoding 2 * encoding_size * hidden_size)
    Output layer (hidden_size * 1)
    """
    def __init__(self, hidden_size, encoding_size, dropout=0.5):
        super(LinearClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.enc2h = nn.Linear(2 * encoding_size, self.hidden_size)
        self.h20 = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, sentence_vecs):
        hidden = self.tanh(self.dropout(self.enc2h(sentence_vecs)))
        out = self.sigmoid(self.h20(hidden))
        return out

class LinearClassifierWithEncoder(nn.Module):
    def __init__(self, hidden_size, encoding_size,
                 embedding_size, num_layers, dropout=0.5,
                 encoder_type="lstm_pooling_classifier",
                 encoder_num_layers=1,
                 encoder_path=None):
        super(LinearClassifierWithEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.encoding_size = encoding_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.encoder_num_layers = encoder_num_layers

        self.model = LinearClassifier(self.hidden_size, self.encoding_size, dropout)
        self.encoder = get_encoder_instance(encoder_type, encoding_size,
                                            embedding_size, encoder_num_layers,
                                            encoder_path)

    def forward(self, x):
        _, encoding = self.encoder(x)
        output = self.model.forward(encoding)
        return output, None

class LinearClassifierWithLM(nn.Module):
    def __init__(self, hidden_size, encoding_size,
                 embedding_size, num_layers, dropout=0.5,
                 encoder_type="LM",
                 encoder_num_layers=1,
                 encoder_path=None):
        super(LinearClassifierWithLM, self).__init__()
        self.hidden_size = hidden_size
        self.encoding_size = encoding_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.encoder_num_layers = encoder_num_layers

        self.model = LinearClassifier(self.hidden_size, self.encoding_size, dropout)
        self.encoder = get_encoder_instance(encoder_type, encoding_size,
                                            embedding_size, encoder_num_layers,
                                            encoder_path)
        with open(self.args.checkpoint, 'rb') as f:
            self.encoder = torch.load(f, map_location=lambda storage, loc: storage)
        self.encoder.eval()

        if self.args.gpu:
            self.encoder.cuda()
        else:
            self.encoder.cpu()

    def forward(self, x):
        _, hiddens = self.encoder(x)
        pool = nn.functional.max_pool1d(hiddens.transpose(1, 2), x.shape[1])
        pool = pool.transpose(1, 2).squeeze()
        output = self.model.forward(pool)
        return output, None



def get_encoder_instance(encoder_type, encoding_size, embedding_size,
                         encoder_num_layers,
                         encoder_path=None):

    encoder = lambda x: x
    if encoder_type == "lstm_pooling_classifier":
        encoder = LSTMPoolingClassifier(
            hidden_size=encoding_size,
            embedding_size=embedding_size,
            num_layers=encoder_num_layers
        )

        if encoder_path is not None:
            encoder.load_state_dict(torch.load(encoder_path)['model'])

            # Since we have loaded freeze params
            for p in encoder.parameters():
                p.requires_grad = False



    return encoder

