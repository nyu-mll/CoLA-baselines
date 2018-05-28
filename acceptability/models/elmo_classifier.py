import torch

from torch import nn


class ELMOClassifier(nn.Module):
    def __init__(self, lm_path, last_hid, dropout=0.5, use_gpu=True):
        super(ELMOClassifier, self).__init__()

        self.lm = torch.load(lm_path)
        self.emb_dim = self.lm.emb_dim
        self.seq_length = self.lm.seq_length
        self.hidden_dim = self.lm.hidden_dim
        self.batch_size = self.lm.batch_size
        self.num_layers = self.lm.num_layers

        self.last_hid = last_hid

        self.lstms = []

        lm_lstm = self.lm.lstm
        lm_state_dict = lm_lstm.state_dict()

        for i in range(self.num_layers):
            # Hacks to get hidden states for each timestep for each layer
            # of LSTM
            if i == 0:
                lstm_inp_dim = self.emb_dim
            else:
                lstm_inp_dim = self.hidden_dim
            lstm = nn.LSTM(lstm_inp_dim, self.hidden_dim, 1,
                           dropout=0, batch_first=True)
            layer_num = str(i)
            ih_weight = lm_state_dict['weight_ih_l' + layer_num]
            hh_weight = lm_state_dict['weight_hh_l' + layer_num]
            ih_bias = lm_state_dict['bias_ih_l' + layer_num]
            hh_bias = lm_state_dict['bias_hh_l' + layer_num]

            curr_state_dict = {
                'weight_ih_l0': ih_weight,
                'bias_ih_l0': ih_bias,
                'weight_hh_l0': hh_weight,
                'bias_hh_l0': hh_bias
            }

            lstm.load_state_dict(curr_state_dict)

            # Freeze LSTM
            for p in lstm.parameters():
                p.requires_grad = False
            self.lstms.append(lstm)

        self.lstms = nn.ModuleList(self.lstms)
        self.dropout = nn.Dropout(p=dropout)
        self.linear_comb = nn.Linear(self.num_layers, 1)
        self.fc1 = nn.Linear(self.hidden_dim, self.last_hid)
        self.relu = nn.ReLU()
        self.out = nn.Linear(self.last_hid, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        hidden_states = []
        hidden = x
        for l in range(self.num_layers):
            hidden, _ = self.lstms[l](hidden)
            # hidden: B x T x H
            hidden_states.append(hidden)

        # [B x T x H] => L x B x T x H
        hidden_states = torch.stack(hidden_states, hidden.dim())
        # L x B x T x H => B x T x H
        hidden_states = self.linear_comb(hidden_states).squeeze()

        num_timesteps = x.shape[1]

        # B x T x H => B x H
        max_pooled = nn.functional.max_pool1d(hidden_states.transpose(1, 2), num_timesteps)
        max_pooled = max_pooled.squeeze()

        non_lineared = self.relu(self.fc1(max_pooled))

        return self.sigmoid(self.out(non_lineared)), hidden_states
