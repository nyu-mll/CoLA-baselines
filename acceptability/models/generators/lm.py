import torch

from torch import nn
from torch.autograd import Variable


class LMGeneratorLSTM(nn.Module):
    """
    LSTM language model used primarily to generate "fake" sentences.
    Could also be used as encoder.
    uses LSTMCell instead of LSTM because forward takes in one time
    step at a time for greedy (or beam search) generation
    """
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super(LMGeneratorLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.ih2h = nn.LSTMCell(input_size, hidden_size)
        self.h2h = nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

        self.log_softmax = nn.functional.log_softmax
        self.softmax = nn.functional.softmax

    def forward(self, input, hidden_states):
        """
        call separately for each time step
        in evaluation or generation mode,
        use output at t-1 to sample input at t
        :param input: batch_size X 1, all tokens of a given time step
        :param hidden_states:
        :return: distribution over vocabulary
        """
        h, c = self.ih2h(input, hidden_states[0])
        next_hiddens = [(h, c)]
        h, c = self.h2h(h, hidden_states[1])
        next_hiddens.append((h, c))
        output = self.log_softmax(self.h2o(h))
        return output, next_hiddens

    def init_hidden(self, batch_size):
        hidden_states = []
        for i in range(self.n_layers + 1):
            hidden_states.append((
                Variable(torch.zeros(batch_size, self.hidden_size)),
                Variable(torch.zeros(batch_size, self.hidden_size))))
        return hidden_states

    def init_hidden_single(self):
        hidden_states = []
        for i in range(self.n_layers + 1):
            hidden_states.append((Variable(torch.zeros(1, self.hidden_size)),
                                  Variable(torch.zeros(1, self.hidden_size))))
        return hidden_states

    def n_params(self):
        return (self.input_size + self.hidden_size) * self.hidden_size + \
            self.n_layers * self.hidden_size * self.hidden_size + \
            self.hidden_size * self.output_size
