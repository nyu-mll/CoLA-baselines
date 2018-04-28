import os
import sys
import nltk
import torch
from torch.utils.data import Dataset
from torchtext import vocab, data
from collections import defaultdict
from acceptability.utils import pad_sentences


class AcceptabilityDataset(Dataset):
    def __init__(self, args, path, vocab):
        self.pairs = []
        self.sentences = []
        self.actual = []
        self.args = args
        if not os.path.exists(path):
            # TODO: log failure here
            sys.exit(1)

        self.vocab = vocab

        with open(path, 'r') as f:
            for line in f:
                line = line.split("\t")

                if len(line) >= 4:
                    self.pairs.append((int(line[1]), line[0]))
                    self.actual.append(line[3])
                    self.sentences.append(self.preprocess(line[3].strip()))

        # TODO: Maybe try later using collate_fn?
        self.sentences, self.sizes = pad_sentences(self.sentences, self.vocab,
                                                   self.args.crop_pad_length)

    def preprocess(self, line):
        tokenizer = lambda x: x
        if not self.args.should_not_preprocess_data:
            if self.args.preprocess_tokenizer == 'nltk':
                tokenizer = nltk_tokenize
            elif self.args.preprocess_tokenizer == 'space':
                tokenizer = lambda x: x.split(' ')

        if not self.args.should_not_lowercase:
            line = line.lower()

        line = tokenizer(line)
        line = [self.vocab.stoi[word] for word in line]

        return line

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        return self.sentences[index], self.pairs[index][0], self.pairs[index][1]


def nltk_tokenize(sentence):
    return nltk.word_tokenize(sentence)

def preprocess_label(label):
    if float(label) > 0:
        return '1'
    else:
        return '0'

def get_datasets(args):
    if args.glove:
        vocab = GloVeIntersectedVocab(args, True)
    else:
        vocab = Vocab(args.vocab_file, True)

    train_dataset = AcceptabilityDataset(args, os.path.join(args.data, 'train.tsv'),
                                            vocab)
    valid_dataset = AcceptabilityDataset(args, os.path.join(args.data, 'valid.tsv'),
                                            vocab)
    test_dataset = AcceptabilityDataset(args, os.path.join(args.data, 'test.tsv'),
                                        vocab)

    return train_dataset, valid_dataset, test_dataset, vocab

class Vocab:
    UNK_TOKEN = '<unk>'
    SOS_TOKEN = '<s>'
    EOS_TOKEN = '</s>'
    PAD_TOKEN = '<pad>'

    UNK_INDEX = 0
    SOS_INDEX = 1
    EOS_INDEX = 2
    PAD_INDEX = 3


    def __init__(self, vocab_path, use_pad=False):
        if not os.path.exists(vocab_path):
            print("Vocab not found at " + vocab_path)
            sys.exit(1)

        self.itos = [''] * 3
        self.itos[self.UNK_INDEX] = self.UNK_TOKEN
        self.itos[self.SOS_INDEX] = self.SOS_TOKEN
        self.itos[self.EOS_INDEX] = self.EOS_TOKEN

        if use_pad:
            self.itos.append(self.PAD_INDEX)

        # Return unk index by default
        self.stoi = defaultdict(lambda: self.UNK_INDEX)
        self.stoi[self.SOS_TOKEN] = self.SOS_INDEX
        self.stoi[self.EOS_TOKEN] = self.EOS_INDEX
        self.stoi[self.UNK_TOKEN] = self.UNK_INDEX

        if use_pad:
            self.stoi[self.PAD_INDEX] = self.PAD_INDEX

        index = len(self.itos)

        with open(vocab_path, 'r') as f:
            for line in f:
                self.itos.append(line.strip())
                self.stoi[line.strip()] = index
                index += 1

    def get_itos(self):
        return self.itos


    def get_stoi(self):
        return self.stoi

    def get_size(self):
        return len(self.itos)


class GloVeIntersectedVocab(Vocab):
    def __init__(self, args, use_pad=True):
        super(GloVeIntersectedVocab, self).__init__(args.vocab_file, use_pad)
        name = args.embedding.split('.')[1]
        dim = args.embedding.split('.')[2][:-1]
        glove = vocab.GloVe(name, int(dim))

        self.vectors = torch.FloatTensor(self.get_size(), len(glove.vectors[0]))
        self.vectors[0].zero_()

        for i in range(1, 4):
            self.vectors[i] = torch.ones_like(self.vectors[i]) * 0.1 * i

        for i in range(4, self.get_size()):
            word = self.itos[i]
            glove_index = glove.stoi.get(word, None)

            if glove_index is None:
                self.vectors[i] = self.vectors[self.UNK_INDEX].copy()
            else:
                self.vectors[i] = glove.vectors[glove_index]

class LMDataset():
    def __init__(self, dataset_path, vocab_path):
        if not os.path.exists(dataset_path):
            print("Dataset not found at " + dataset_path)
            sys.exit(1)

        self.vocab = Vocab(vocab_path)

        num_tokens = 0
        with open(dataset_path, 'r') as f:
            for line in f:
                line = line.split("\t")
                if len(line) >= 4:
                    words = self.preprocess(line[3].split(' '))
                    num_tokens += len(words)

        self.tokens = torch.LongTensor(num_tokens)

        num_tokens = 0
        with open(dataset_path, 'r') as f:
            for line in f:
                line = line.split("\t")

                if len(line) >= 4:
                    words = self.preprocess(line[3].strip().split(' '))
                    for word in words:
                        self.tokens[num_tokens] = self.vocab.stoi[word]
                        num_tokens += 1


    def get_vocab_size(self):
        return self.vocab.get_size()

    def preprocess(self, x):
        return [self.vocab.SOS_TOKEN] + x + [self.vocab.EOS_TOKEN]

    def get_tokens(self):
        return self.tokens



class LMEvalDataset():
    def __init__(self, dataset_path, vocab_path):
        if not os.path.exists(dataset_path):
            print("Dataset not found at " + dataset_path)
            sys.exit(1)

        self.vocab = Vocab(vocab_path)
        self.sentences = []

        with open(dataset_path, 'r') as f:
            for line in f:
                line = line.split("\t")

                if len(line) >= 4:
                    words = self.preprocess(line[3].strip().split(' '))
                    self.sentences.append([self.vocab.stoi[x] for x in words])

        # # TODO: Maybe try later using collate_fn?
        # self.sentences, self.sizes = pad_sentences(self.sentences, self.vocab)


    def get_vocab_size(self):
        return self.vocab.get_size()

    def preprocess(self, x):
        return [self.vocab.SOS_TOKEN] + x + [self.vocab.EOS_TOKEN]

