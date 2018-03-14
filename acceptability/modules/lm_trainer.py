import torch
import os
import sys
import numpy as np

from torch.autograd import Variable
from acceptability.utils import get_lm_parser, get_lm_model_instance, get_lm_experiment_name
from acceptability.utils import Checkpoint, Timer
from .dataset import LMDataset
from .early_stopping import EarlyStopping
from .logger import Logger


class LMTrainer:
    def __init__(self):
        parser = get_lm_parser()
        self.args = parser.parse_args()
        self.train_data = LMDataset(self.args.file, self.args.vocab_file,
                                    self.args.seq_length)
        self.args.vocab_size = self.train_data.get_vocab_size()
        self.args.gpu = self.args.gpu and torch.cuda.is_available()
        self.train_loader = torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.args.batch_size,
            shuffle=True
        )

        if self.args.experiment_name is None:
            self.args.experiment_name = get_lm_experiment_name(self.args)
        self.checkpoint = Checkpoint(self.args)
        self.writer = Logger(self.args)
        self.writer.write(self.args)
        self.timer = Timer()

    def load(self):
        self.model = get_lm_model_instance(self.args)

        if self.model is None:
            # TODO: Add logger statement for valid model here
            sys.exit(1)

        self.checkpoint.load_state_dict(self.model)

        if self.args.gpu:
            self.model = self.model.cuda()

        self.early_stopping = EarlyStopping(self.model, self.checkpoint)

        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                                 self.model.parameters()),
                                          lr=self.args.learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()

# Truncated Backpropagation
    def detach(self, states):
        return [state.detach() for state in states]

    def train(self):
        self.model.train()
        total_loss = 0
        vocab_size = self.args.vocab_size
        hidden = self.model.init_hidden()
        for epoch in range(1, self.args.epochs + 1):
            for idx, vals in enumerate(self.train_loader):
                data, target = vals
                data, target = Variable(data), Variable(target)
                hidden = self.detach(hidden)
                self.model.zero_grad()

                output, hidden = self.model(data, hidden)

                loss = self.criterion(output, target.view(-1))
                loss.backward()

                torch.nn.utils.clip_grad_norm(self.model.parameters(),
                                              self.args.clip)

                total_loss += loss.data

                step = (idx + 1) // self.args.seq_length
                if step % 100 == 0:
                    total_loss = 0
                    print('Epoch [%d/%d], Step[%d/%d], Loss: %.3f, Perplexity: %5.2f' %
                            (epoch+1, self.args.epochs, step, len(self.train_loader),
                             loss.data[0], np.exp(loss.data[0])))
