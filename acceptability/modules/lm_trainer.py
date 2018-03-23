import torch
import os
import sys
import math
import numpy as np

from torch.autograd import Variable
from acceptability.utils import get_lm_parser, get_lm_model_instance, get_lm_experiment_name
from acceptability.utils import Checkpoint, Timer
from acceptability.utils import batchify, get_batch, repackage_hidden
from .dataset import LMDataset
from .early_stopping import EarlyStopping
from .logger import Logger


class LMTrainer:
    def __init__(self):
        parser = get_lm_parser()
        self.args = parser.parse_args()
        print("Loading datasets")
        self.train_data = LMDataset(os.path.join(self.args.data, 'train.tsv'), self.args.vocab_file,
                                    self.args.seq_length)
        print("Train dataset loaded")
        self.val_data = LMDataset(os.path.join(self.args.data, 'valid.tsv'), self.args.vocab_file,
                                  self.args.seq_length)
        print("Val dataset loaded")

        self.args.vocab_size = self.train_data.get_vocab_size()
        self.args.gpu = self.args.gpu and torch.cuda.is_available()


        print("Created dataloaders")
        self.train_loader = batchify(self.train_data.get_tokens(), self.args.batch_size,
                                     self.args)
        self.val_loader = batchify(self.val_data.get_tokens(), self.args.batch_size,
                                     self.args)

        if self.args.experiment_name is None:
            self.args.experiment_name = get_lm_experiment_name(self.args)
        self.checkpoint = Checkpoint(self.args)
        self.writer = Logger(self.args)
        self.writer.write(self.args)
        self.timer = Timer()

    def load(self):
        print("Creating model instance")
        self.model = get_lm_model_instance(self.args)

        if self.model is None:
            self.writer.write("Please pass a valid model name")
            sys.exit(1)

        self.checkpoint.load_state_dict(self.model)

        if self.args.gpu:
            self.model = self.model.cuda()

        self.early_stopping = EarlyStopping(self.model, self.checkpoint, minimize=True)

        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                                 self.model.parameters()),
                                          lr=self.args.learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()

    # Truncated Backpropagation
    def detach(self, states):
        return [state.detach() for state in states]

    def train(self):
        print("Starting training")
        self.model.train()
        self.print_start_info()
        log_interval = len(self.train_loader) // self.args.stages_per_epoch

        for epoch in range(1, self.args.epochs + 1):
            total_loss = 0
            ntokens = self.train_data.get_vocab_size()
            hidden = self.model.init_hidden(self.args.batch_size)

            for step, i in enumerate(range(0, self.train_loader.size(0) -1, self.args.seq_length)):
                data, targets = get_batch(self.train_loader, i, self.args.seq_length)
                hidden = repackage_hidden(hidden)

                self.model.zero_grad()

                output, hidden = self.model(data, hidden)

                loss = self.criterion(output.view(-1, ntokens), targets)
                loss.backward()

                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.args.clip)
                self.optimizer.step()

                total_loss += loss.data

                if step % log_interval == 0 and step > 0:
                    curr_loss = total_loss[0] / log_interval
                    self.writer.write(
                        'Train: Epoch [%d/%d], Step[%d/%d], Loss: %.3f, Perplexity: %5.2f' %
                            (epoch, self.args.epochs, step, len(self.train_loader) // self.args.seq_length,
                             curr_loss, math.exp(curr_loss)))
                    total_loss = 0

                    val_loss = self.validate()
                    stop = self.early_stopping(np.exp(val_loss), {'val_loss': val_loss}, epoch)

                    if stop:
                        self.writer.write("Early Stopping activated")
                        break
                    else:
                        self.writer.write(
                            'Val: Epoch [%d/%d], Step[%d/%d], Loss: %.3f, Perplexity: %5.2f' %
                                (epoch, self.args.epochs, step, len(self.train_loader),
                                val_loss, np.exp(val_loss)))

            if self.early_stopping.is_activated():
                break
            self.print_epoch_info()


    def validate(self):
        self.model.eval()
        total_loss = 0
        hidden = self.model.init_hidden(self.args.batch_size)
        ntokens = self.train_data.get_vocab_size()

        for batch, i in enumerate(range(0, self.val_loader.size(0) - 1, self.args.seq_length)):
            data, targets = get_batch(self.val_loader, i, self.args.seq_length, evaluation=True)
            output, hidden = self.model(data, hidden)
            output_flat = output.view(-1, ntokens)

            total_loss += len(data) * self.criterion(output_flat, targets).data
            hidden = repackage_hidden(hidden)

        self.model.train()

        return total_loss[0] / len(self.val_loader)

    def print_epoch_info(self):
        self.writer.write_new_line()
        self.writer.write(self.early_stopping.get_info_lm())
        self.writer.write("Time Elasped: %s" % self.timer.get_current())

    def print_start_info(self):
        self.writer.write("======== General =======")
        self.writer.write("Model: %s" % self.args.model)
        self.writer.write("GPU: %s" % self.args.gpu)
        self.writer.write("Experiment Name: %s" % self.args.experiment_name)
        self.writer.write("Save location: %s" % self.args.save_loc)
        self.writer.write("Logs dir: %s" % self.args.logs_dir)
        self.writer.write("Timestamp: %s" % self.timer.get_time_hhmmss())
        self.writer.write_new_line()

        self.writer.write("======== Data =======")
        self.writer.write("Training set: %d examples of size %d" %
                          (len(self.train_data.get_tokens()) // self.args.seq_length,
                           self.args.seq_length))
        self.writer.write("Validation set: %d examples of size %d" %
                          (len(self.val_data.get_tokens()) // self.args.seq_length,
                           self.args.seq_length))
        self.writer.write_new_line()

        self.writer.write("======= Parameters =======")
        self.writer.write("Vocab Size: %d" % self.args.vocab_size)
        self.writer.write("Sequence Length: %d" % self.args.seq_length)
        self.writer.write("Learning Rate: %f" % self.args.learning_rate)
        self.writer.write("Batch Size: %d" % self.args.batch_size)
        self.writer.write("Epochs: %d" % self.args.epochs)
        self.writer.write("Patience: %d" % self.args.patience)
        self.writer.write("Stages per Epoch: %d" % self.args.stages_per_epoch)
        self.writer.write("Embedding: %s" % self.args.embedding_size)
        self.writer.write("Number of layers: %d" % self.args.num_layers)
        self.writer.write("Hidden Size: %d" % self.args.hidden_size)
        self.writer.write("Resume: %s" % self.args.resume)
        self.writer.write_new_line()

        self.writer.write("======= Model =======")
        self.writer.write(self.model)
        self.writer.write_new_line()


