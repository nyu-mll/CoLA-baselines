import argparse
import torch
import os

from torch.autograd import Variable

from .dataset import Vocab
from acceptability.utils import seed_torch
from acceptability.utils.flags import get_lm_evaluator_parser
from .dataset import LMEvalDataset
from acceptability.utils import batchify, get_batch, repackage_hidden
import numpy as np
import torch.nn.functional as F




class LMEvaluator():
    def __init__(self):
        parser = get_lm_evaluator_parser()
        self.args = parser.parse_args()
        print(self.args)
        seed_torch(self.args)
        self.vocab = Vocab(self.args.vocab_file)
        self.ntokens = self.vocab.get_size()
        self.data = LMEvalDataset(os.path.join(self.args.data), self.args.vocab_file)
        self.out = open(self.args.outf, "w")


    def load(self):
        with open(self.args.checkpoint, 'rb') as f:
            self.model = torch.load(f, map_location=lambda storage, loc: storage)
        self.model.eval()

        if self.args.gpu:
            self.model.cuda()
        else:
            self.model.cpu()

    def get_batches(self, loader):
        batches = []
        for _, i in enumerate(range(0, loader.size(0) -1, self.args.seq_length)):
            data, targets = get_batch(loader, i, self.args.seq_length)
            batches.append((data, targets))
        return batches

    def eval(self):
        for i in self.data.sentences:
            hidden = self.model.init_hidden(1)
            data = Variable(torch.LongTensor(i))
            hidden = repackage_hidden(hidden)
            output, hiddens = self.model(data, hidden)
            output = F.log_softmax(output)
            log_probs = []
            for w, o in zip(i[1:], output[:-1]):
                log_probs.append(o[w])
            self.out.write("%s,%s\n" % (str(sum(log_probs).data[0]), ",".join([str(x.data[0]) for x in log_probs])))
        self.out.close()


