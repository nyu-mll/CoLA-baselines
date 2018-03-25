import argparse
import torch
import os

from torch.autograd import Variable

from .dataset import LMDataset
from acceptability.utils import get_lm_generator_parser, seed_torch


class LMGenerator():
    def __init__(self):
        parser = get_lm_generator_parser()
        self.args = parser.parse_args()
        print(self.args)
        if self.args.temperature < 1e-3:
            parser.error("--temperature has to be greater or equal 1e-3")

        seed_torch(self.args)


    def load(self):
        with open(self.args.checkpoint, 'rb') as f:
            self.model = torch.load(f)
        self.model.eval()

        if self.args.gpu:
            self.model.cuda()
        else:
            self.model.cpu()

        self.corpus = LMDataset(os.path.join(self.args.data, 'valid.tsv'), self.args.vocab_file)
        self.ntokens = self.corpus.get_vocab_size()

    def generate(self):
        hidden = self.model.init_hidden(1)
        inp = Variable(torch.LongTensor([self.corpus.SOS_INDEX]).unsqueeze(0), volatile=True)
        if self.args.gpu:
            inp.data = inp.data.cuda()

        with open(self.args.outf, 'w') as outf:
            for i in range(self.args.nlines):
                words = []
                while True:
                    output, hidden = self.model(inp, hidden)
                    word_weights = output.squeeze().data.div(self.args.temperature).exp().cpu()
                    word_idx = torch.multinomial(word_weights, 1)[0]
                    inp.data.fill_(word_idx)
                    word = self.corpus.itos[word_idx]

                    if word == '</s>':
                        line = ['lm', '0', '', ' '.join(words) + '\n']
                        outf.write('\t'.join(line))
                        break
                    else:
                        words.append(word)

                if i % self.args.log_interval == 0:
                    print('| Generated {}/{} lines, {} words'
                            .format(i, self.args.nlines, len(words)))
