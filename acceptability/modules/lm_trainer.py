import torch
import os
import sys
import math
import numpy as np

from torch.autograd import Variable
from acceptability.utils import get_lm_parser, get_lm_model_instance, get_lm_experiment_name
from acceptability.utils import Checkpoint, Timer
from acceptability.utils import batchify, get_batch, repackage_hidden
from acceptability.utils import seed_torch
from .dataset import LMDataset
from .early_stopping import EarlyStopping
from .logger import Logger


class LMTrainer:
    def __init__(self):
        parser = get_lm_parser()
        self.args = parser.parse_args()

        seed_torch(self.args)
        print("Loading datasets")
        self.train_data = LMDataset(os.path.join(self.args.data, 'train.tsv'), self.args.vocab_file)
        print("Train dataset loaded")

        self.val_data = LMDataset(os.path.join(self.args.data, 'valid.tsv'), self.args.vocab_file)
        print("Val dataset loaded")

        self.test_data = LMDataset(os.path.join(self.args.data, 'test.tsv'), self.args.vocab_file)
        print("Test dataset loaded")

        self.args.vocab_size = self.train_data.get_vocab_size()
        self.args.gpu = self.args.gpu and torch.cuda.is_available()


        print("Created dataloaders")
        self.train_loader = batchify(self.train_data.get_tokens(), self.args.batch_size,
                                     self.args)
        self.val_loader = batchify(self.val_data.get_tokens(), self.args.batch_size,
                                   self.args)

        self.test_loader = batchify(self.test_data.get_tokens(), self.args.batch_size,
                                    self.args)

        if self.args.experiment_name is None:
            self.args.experiment_name = get_lm_experiment_name(self.args)
        self.checkpoint = Checkpoint(self)
        self.writer = Logger(self.args)
        self.writer.write(self.args)
        self.timer = Timer()

    def load(self):
        print("Creating model instance")
        self.model = get_lm_model_instance(self.args)

        if self.model is None:
            self.writer.write("Please pass a valid model name")
            sys.exit(1)

        self.current_epoch = 0

        self.early_stopping = EarlyStopping(self.model, self.checkpoint, patience=self.args.patience,
                                            minimize=True)

        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                                 self.model.parameters()),
                                          lr=self.args.learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.checkpoint.load_state_dict()

        if torch.cuda.device_count() > 1:
            print("Using ", torch.cuda.device_count(), " GPUs")
            self.model = torch.nn.DataParallel(self.model)

        if self.args.gpu:
            self.model = self.model.cuda()

    # Truncated Backpropagation
    def detach(self, states):
        return [state.detach() for state in states]

    def get_batches(self, loader):
        batches = []
        for _, i in enumerate(range(0, loader.size(0) -1, self.args.seq_length)):
            data, targets = get_batch(loader, i, self.args.seq_length)
            batches.append((data, targets))

        return batches

    def train(self):
        print("Starting training")
        self.model.train()
        self.log_interval = len(self.train_loader) // self.args.stages_per_epoch
        self.log_interval //= self.args.seq_length

        if self.log_interval <= 0:
            self.log_interval = 1

        self.print_start_info()

        batches = self.get_batches(self.train_loader)

        for epoch in range(self.current_epoch + 1, self.args.epochs + 1):
            self.current_epoch = epoch
            total_loss = 0
            ntokens = self.train_data.get_vocab_size()

            if type(self.model) == torch.nn.DataParallel:
                hidden = self.model.module.init_hidden(self.args.batch_size)
            else:
                hidden = self.model.init_hidden(self.args.batch_size)

            for step, i in enumerate(np.random.permutation(len(batches))):
                data, targets = batches[i]
                data, targets = Variable(data), Variable(targets)
                hidden = repackage_hidden(hidden)

                self.model.zero_grad()

                output, hidden = self.model(data, hidden)

                loss = self.criterion(output.view(-1, ntokens), targets)
                loss.backward()

                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.args.clip)
                self.optimizer.step()

                total_loss += loss.data

                if (step + 1) % self.log_interval == 0 and step > 0:
                    curr_loss = total_loss[0] / self.log_interval
                    self.writer.write(
                        'Train: Epoch [%d/%d], Step[%d/%d], Loss: %.3f, Perplexity: %5.2f' %
                            (epoch, self.args.epochs, step, len(self.train_loader) // self.args.seq_length,
                             curr_loss, math.exp(curr_loss)))
                    total_loss = 0

                    val_loss = self.validate(self.val_loader)
                    stop = self.early_stopping(math.exp(val_loss), {'val_loss': val_loss}, epoch)

                    if stop:
                        self.writer.write("Early Stopping activated")
                        break
                    else:
                        self.writer.write(
                            'Val: Epoch [%d/%d], Step[%d/%d], Loss: %.3f, Perplexity: %5.2f' %
                                (epoch, self.args.epochs, step + 1, len(self.train_loader) // self.args.seq_length,
                                val_loss, math.exp(val_loss)))

            if self.early_stopping.is_activated():
                break
            self.print_epoch_info()

        self.checkpoint.restore()
        self.checkpoint.finalize()

    def validate(self, loader):
        self.model.eval()
        total_loss = 0

        if type(self.model) == torch.nn.DataParallel:
            hidden = self.model.module.init_hidden(self.args.batch_size)
        else:
            hidden = self.model.init_hidden(self.args.batch_size)

        ntokens = self.train_data.get_vocab_size()

        tokens = 0

        for _, i in enumerate(range(0, loader.size(0) - 1, self.args.seq_length)):
            data, targets = get_batch(loader, i, self.args.seq_length, evaluation=True)
            data, targets = Variable(data, volatile=True), Variable(targets, volatile=True)
            output, hidden = self.model(data, hidden)
            output_flat = output.view(-1, ntokens)
            tokens += output_flat.size(0)
            total_loss += self.criterion(output_flat, targets).data
            hidden = repackage_hidden(hidden)

        self.model.train()
        return total_loss[0] / (loader.size(0) / self.args.seq_length)

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
        self.writer.write("Log Interval: %s" % self.log_interval)
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


