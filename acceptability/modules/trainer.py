import torch
import os
import sys
import torchtext
import numpy as np

from torch import nn
from torch.autograd import Variable
from acceptability.utils import get_parser, get_model_instance, get_experiment_name
from acceptability.utils import seed_torch, pad_sentences
from .dataset import get_datasets
from .meter import Meter
from .early_stopping import EarlyStopping
from .logger import Logger
from acceptability.utils import Checkpoint
from acceptability.utils import Timer


class Trainer:
    def __init__(self):
        parser = get_parser()
        self.args = parser.parse_args()
        self.args.gpu = self.args.gpu and torch.cuda.is_available()

        seed_torch(self.args)

        if self.args.experiment_name is None:
            self.args.experiment_name = get_experiment_name(self.args)
        self.checkpoint = Checkpoint(self)
        self.num_classes = 2
        self.meter = Meter(self.num_classes)
        self.writer = Logger(self.args)
        self.writer.write(self.args)
        self.timer = Timer()
        self.load_datasets()

        if self.args.imbalance is not None:
            self.weights = np.array([self.args.imbalance, 1 - self.args.imbalance])
        else:
            self.weights = None

    def load_datasets(self):
        self.train_dataset, self.val_dataset, self.test_dataset, \
            vocab = get_datasets(self.args)

        if self.args.glove:
            self.vocab = vocab
            self.embedding = nn.Embedding(len(vocab.vectors), len(vocab.vectors[0]))
            self.embedding.weight.data.copy_(vocab.vectors)
        else:
            self.vocab = vocab

            self.embedding = nn.Embedding(self.vocab.get_size(), self.args.embedding_size)
            self.embedding.weight.data.uniform_(-0.1, 0.1)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            pin_memory=self.args.gpu
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            pin_memory=self.args.gpu
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.args.batch_size,
            pin_memory=self.args.gpu
        )

        if not self.args.train_embeddings:
            self.embedding.weight.requires_grad = False
            self.embedding.eval()

    def load(self):
        self.model = get_model_instance(self.args)

        if self.model is None:
            print("model not found at " + self.args.model)
            sys.exit(1)

        self.early_stopping = EarlyStopping(self.model, self.checkpoint, self.args.patience)

        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                                 self.model.parameters()),
                                          lr=self.args.learning_rate)
        self.criterion = torch.nn.BCELoss()

        self.current_epoch = 0
        self.checkpoint.load_state_dict()

        if self.args.gpu:
            self.model = self.model.cuda()
            self.embedding = self.embedding.cuda()


    def train(self):
        self.print_start_info()
        log_interval = len(self.train_loader) // self.args.stages_per_epoch

        if log_interval <= 0:
            log_interval = 1

        for i in range(self.current_epoch + 1, self.args.epochs + 1):
            self.current_epoch = i
            self.writer.write("========= Epoch %d =========" % i)

            for idx, data in enumerate(self.train_loader):
                x, y, _ = data
                x, y = Variable(x).long(), Variable(y)

                if self.args.gpu:
                    x = x.cuda()
                    y = y.cuda()

                x = self.embedding(x)

                self.optimizer.zero_grad()

                output = self.model(x)

                if type(output) == tuple:
                    output = output[0]
                output = output.squeeze()

                if self.weights is not None:
                    weights = torch.from_numpy(self.weights[y.data.cpu().numpy()])
                    weights = weights.float()

                    if self.args.gpu:
                        weights = weights.cuda()

                    loss = nn.functional.binary_cross_entropy(output, y.float(),
                                                              weight=weights)
                else:
                    loss = self.criterion(output, y.float())
                loss.backward()

                self.optimizer.step()

                if (idx + 1) % log_interval == 0 and idx > 0:
                    acc, loss, matthews, confusion = self.validate(self.val_loader)
                    other_metrics = {
                        'acc': acc,
                        'val_loss': loss,
                        'confusion_matrix': confusion
                    }
                    stop = self.early_stopping(matthews, other_metrics, i)
                    if stop:
                        self.writer.write("Early Stopping activated")
                        break
                    else:
                        self.print_current_info(idx + 1, len(self.train_loader),
                                                matthews, other_metrics)


            self.print_epoch_info()

            if self.args.evaluate_train and i % self.args.train_evaluate_interval == 0:
                # At the some interval validate train loader
                self.writer.write("Evaluating training set")
                acc, loss, matthews, confusion = self.validate(self.train_loader)
                other_metrics = {
                    'acc': acc,
                    'val_loss': loss,
                    'confusion_matrix': confusion
                }
                self.writer.write("Epoch:")
                self.print_current_info(i, self.args.epochs, matthews, other_metrics)

            if self.early_stopping.is_activated():
                break

        self.checkpoint.restore()
        acc, loss, matthews, confusion = self.validate(self.test_loader, test=True)
        other_metrics = {
            'acc': acc,
            'val_loss': loss,
            'confusion_matrix': confusion
        }
        self.writer.write("Test Set:")
        self.print_current_info(0, 0, matthews, other_metrics)
        self.checkpoint.finalize()

    def validate(self, loader: torch.utils.data.DataLoader, test=False):
        self.model.eval()
        self.embedding.eval()
        self.meter.reset()
        correct = 0
        total = 0
        total_loss = 0
        outputs = []

        for data in loader:
            x, y, _ = data
            x, y = Variable(x).long(), Variable(y)

            if self.args.gpu:
                x = x.cuda()
                y = y.cuda()

            x = self.embedding(x)

            output = self.model(x)

            if type(output) == tuple:
                output = output[0]
            output = output.squeeze()

            if self.weights is not None:
                weights = torch.from_numpy(self.weights[y.data.cpu().numpy()])
                weights = weights.float()
                if self.args.gpu:
                    weights = weights.cuda()

                loss = nn.functional.binary_cross_entropy(output, y.float(),
                                                          weight=weights,
                                                          size_average=False)
            else:
                loss = nn.functional.binary_cross_entropy(output, y.float(),
                                                          size_average=False)
            total_loss = loss.data[0]
            total += len(y)
            output = (output > 0.5).long()
            outputs.extend([int(o) for o in output])

            self.meter.add(output.data, y.data)
            if not self.args.gpu:
                correct += (y ==
                            output).data.cpu().numpy().sum()
            else:
                correct += (y == output).data.sum()
        self.model.train()

        if self.args.train_embeddings:
            self.embedding.train()

        avg_loss = total_loss / total

        if test and self.args.output_dir is not None:
            out_file = open(os.path.join(self.args.output_dir, self.args.experiment_name + ".tsv"), "w")
            for x in outputs:
                out_file.write(str(x) + "\n")
            out_file.close()

        return correct / total * 100, avg_loss, \
               self.meter.matthews(), self.meter.confusion()

    def print_epoch_info(self):
        self.writer.write_new_line()
        self.writer.write(self.early_stopping.get_info())
        self.writer.write("Time Elasped: %s" % self.timer.get_current())

    def print_current_info(self, it, total, matthews, other_metrics):
        self.writer.write("%d/%d: Matthews %.5f, Accuracy: %.5f, Loss: %.9f" %
              (it, total, matthews,
               other_metrics['acc'], other_metrics['val_loss']))

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
        self.writer.write("Training set: %d examples" % (len(self.train_dataset)))
        self.writer.write("Validation set: %d examples" % (len(self.val_dataset)))
        self.writer.write("Test set: %d examples" % (len(self.test_dataset)))
        self.writer.write_new_line()

        self.writer.write("======= Parameters =======")
        self.writer.write("Learning Rate: %f" % self.args.learning_rate)
        self.writer.write("Batch Size: %d" % self.args.batch_size)
        self.writer.write("Epochs: %d" % self.args.epochs)
        self.writer.write("Patience: %d" % self.args.patience)
        self.writer.write("Stages per Epoch: %d" % self.args.stages_per_epoch)

        if self.args.glove:
            self.writer.write("Embedding: %s" % self.args.embedding)
        else:
            self.writer.write("Embedding: %d x %d" % self.embedding.weight.size())
        self.writer.write("Number of layers: %d" % self.args.num_layers)
        self.writer.write("Hidden Size: %d" % self.args.hidden_size)
        self.writer.write("Encoder Size: %d" % self.args.encoding_size)
        self.writer.write("Resume: %s" % self.args.resume)
        self.writer.write_new_line()

        self.writer.write("======= Model =======")
        self.writer.write(self.model)
        self.writer.write_new_line()
