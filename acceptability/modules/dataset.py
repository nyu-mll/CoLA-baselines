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
        return get_datasets_glove(args)
    else:
        vocab = Vocab(args.vocab_file, True)
        train_dataset = AcceptabilityDataset(args, os.path.join(args.data, 'train.tsv'),
                                             vocab)
        valid_dataset = AcceptabilityDataset(args, os.path.join(args.data, 'valid.tsv'),
                                             vocab)
        test_dataset = AcceptabilityDataset(args, os.path.join(args.data, 'test.tsv'),
                                            vocab)

        return train_dataset, valid_dataset, test_dataset, vocab


def get_datasets_glove(args):
    tokenizer = lambda x: x
    if not args.should_not_preprocess_data:
        if args.preprocess_tokenizer == 'nltk':
            tokenizer = nltk_tokenize
        elif args.preprocess_tokenizer == 'space':
            tokenizer = lambda x: x.split(' ')

    sentence = data.Field(
        sequential=True,
        fix_length=args.crop_pad_length,
        tokenize=tokenizer,
        tensor_type=torch.cuda.LongTensor if args.gpu else torch.LongTensor,
        lower=not args.should_not_lowercase,
        batch_first=True
    )

    train_dataset, val_dataset, test_dataset = data.TabularDataset.splits(
        path=args.data,
        train="train.tsv",
        validation="valid.tsv",
        test="test.tsv",
        format="tsv",
        fields=[
            ('source', data.field.RawField()),
            ('label', data.field.LabelField(use_vocab=False,
                                            preprocessing=preprocess_label)),
            ('mark', None),
            ('sentence', sentence)
        ]
    )

    sentence.build_vocab(
        train_dataset,
        vectors=args.embedding
    )

    return train_dataset, val_dataset, test_dataset, sentence

def get_iter(args, dataset):
    return data.Iterator(
        dataset,
        batch_size=args.batch_size,
        device=0 if args.gpu else -1,
        repeat=False
    )

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
