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

        self.embedding = self.lm.embedding
        self.last_hid = last_hid


        # Freeze embedding
        for p in self.embedding.parameters():
            p.requires_grad = False

        self.lstms = []

        lm_lstm = self.lm.lstm
        lm_state_dict = lm_lstm.state_dict()

        for l in range(self.num_layers):
            # Hacks to get hidden states for each timestep for each layer
            # of LSTM
            lstm = nn.LSTM(self.emb_dim, self.hidden_dim, 1, dropout=0)
            ih_weight = lm_state_dict['weight_ih_l' + l]
            hh_weight = lm_state_dict['weight_hh_l' + l]
            ih_base = lm_state_dict['base_ih_l' + l]
            hh_base = lm_state_dict['base_hh_l' + l]

            curr_state_dict = {
                'ih_weight_l0': ih_weight,
                'ih_base_l0': ih_base,
                'hh_weight_l0': hh_weight,
                'hh_base_l0': hh_base
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
        x = self.embedding(x)

        hidden_states = []
        hidden = x
        for l in self.num_layers:
            hidden, _ = self.lstms[l](hidden)
            # hidden: T x B x H
            hidden_states.append(hidden)

        hidden_states = torch.cat(hidden_states, hidden.dim())
        hidden_states = self.linear_comb(hidden_states).squeeze()

        max_pooled = torch.nn.max_pool1d(hidden_states, dim=0).squeeze()

        non_lineared = self.relu(self.fc1(max_pooled))

        return self.sigmoid(self.out(non_lineared))
