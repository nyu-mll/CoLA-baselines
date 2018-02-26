import torch
import os
import sys
import torchtext

from torch import nn
from utils import get_parser
from utils import get_model_instance
from .dataset import get_datasets, get_iter
from .meter import Meter
from .early_stopping import EarlyStopping
from utils import Checkpoint

# TODO: Add __init__ for all modules and then __all__ in all of them
# to faciliate easy loading


class Trainer:
    def __init__(self):
        parser = get_parser()
        self.args = parser.parse_args()
        self.args.gpu = self.args.gpu and torch.cuda.is_available()
        self.checkpoint = Checkpoint(self.args)
        self.num_classes = 2
        self.meter = Meter(self.num_classes)
        self.load_datasets()

    def load_datasets(self):
        self.train_dataset, self.val_dataset, self.test_dataset, \
            sentence_field = get_datasets(self.args)

        self.train_loader = get_iter(self.args, self.train_dataset)
        self.val_loader = get_iter(self.args, self.val_dataset)
        self.test_loader = get_iter(self.args, self.test_dataset)

        vocab = sentence_field.vocab
        self.embedding = nn.Embedding(len(vocab), len(vocab.vectors[0]))
        self.embedding.weight.data.copy_(vocab.vectors)
        self.embedding.weight.requires_grad = False

    def load(self):
        self.model = get_model_instance(self.args)

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
        self.criterion = torch.nn.BCELoss()

    def train(self):
        for i in range(1, self.args.epochs + 1):
            print("========= Epoch %d =========" % i)
            self.train_loader.init_epoch()
            for idx, data in enumerate(self.train_loader):
                x, y = data.sentence, data.label
                x = self.embedding(x)

                self.optimizer.zero_grad()

                output = self.model(x)

                if type(output) == tuple:
                    output = output[0]

                loss = self.criterion(output, y.float())
                loss.backward()

                self.optimizer.step()

                if idx % self.args.stages_per_epoch == 0:
                    # TODO: Validate here
                    # TODO: Add early stopping based on matthews here
                    acc, loss, matthews, confusion = self.validate(self.val_loader)
                    other_metrics = {
                        'acc': acc,
                        'val_loss': loss,
                        'confusion_matrix': confusion
                    }
                    stop = self.early_stopping(matthews, other_metrics, i)
                    if stop:
                        self.print_final_info()
                        break
                    else:
                        self.print_current_info(idx, len(self.train_loader),
                                                matthews, other_metrics)

            if i % 10 == 0:
                # At the some interval validate train loader
                self.validate(self.train_loader)

            self.early_stopping.print_info()

    def print_final_info(self):
        print("Early Stopping activated")
        self.early_stopping.print_info()

    def print_current_info(self, it, total, matthews, other_metrics):
        print("%d/%d: Matthews %.5f, Accuracy: %.5f, Loss: %.9f" %
              (it, total, matthews,
               other_metrics['acc'], other_metrics['val_loss']))


    def validate(self, loader: torchtext.data.Iterator):
        self.model.eval()
        self.meter.reset()
        correct = 0
        total = 0
        total_loss = 0
        loader.init_epoch()
        for idx, data in enumerate(loader):
            x, y = data.sentence, data.label
            x = self.embedding(x)
            output = self.model(x)

            if type(output) == tuple:
                output = output[0]

            loss = nn.functional.binary_cross_entropy(output, y.float(),
                                                      size_average=False)
            total_loss = loss.data[0]
            total += len(y)
            output = (output > 0.5).long()

            self.meter.add(output.data, y.data)
            if not self.args.gpu:
                correct += (y ==
                            output).data.cpu().numpy().sum()
            else:
                correct += (y == output).data.sum()
        self.model.train()

        avg_loss = total_loss / total

        return correct / total * 100, avg_loss, \
               self.meter.matthews(), self.meter.confusion()
