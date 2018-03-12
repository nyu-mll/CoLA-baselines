import os
import sys
import nltk
import torch
from torch.utils.data import Dataset
from torchtext import vocab, data
from collections import defaultdict


class AcceptabilityDataset(Dataset):
    def __init__(self, path, vocab_name):
        self.pairs = []
        self.sentences = []
        if not os.path.exists(path):
            # TODO: log failure here
            sys.exit(1)

        self.vocab = vocab.pretrained_aliases[vocab_name]
        with open(path, 'r') as f:
            for line in f:
                line = line.split("\t")

                if len(line) >= 4:
                    self.sentences.append(line[3].strip())
                    self.pairs.append((line[3].strip(), line[1], line[0]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        return self.pairs[index]


def nltk_tokenize(sentence):
    return nltk.word_tokenize(sentence)

def preprocess_label(label):
    if float(label) > 0:
        return '1'
    else:
        return '0'

def get_datasets(args):
    tokenizer = lambda x: x
    if args.preprocess_data:
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
        path=args.data_dir,
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

class LMDataset(Dataset):
    UNK_TOKEN = '<unk>'
    SOS_TOKEN = '<s>'
    EOS_TOKEN = '</s>'

    UNK_INDEX = 0
    SOS_INDEX = 1
    EOS_INDEX = 2

    def __init__(self, dataset_path, vocab_path):
        super(LMDataset, self).__init__()
        self.sentences = []
        if not os.path.exists(dataset_path):
            sys.exit(1)

        self.itos = [''] * 3
        self.itos[self.UNK_INDEX] = self.UNK_TOKEN
        self.itos[self.SOS_INDEX] = self.SOS_TOKEN
        self.itos[self.EOS_INDEX] = self.EOS_TOKEN

        with open(dataset_path, 'r') as f:
            for line in f:
                line = line.split("\t")

                if len(line) >= 4:
                    self.sentences.append(line[3].strip())

        # Return unk index by default
        self.stoi = defaultdict(self.UNK_INDEX)
        index = 3
        with open(vocab_path, 'r') as f:
            for line in f:
                self.itos.append(line.strip())
                self.stoi[line.strip()] = index
                index += 1

    def get_vocab_size(self):
        return len(self.itos)

    def preprocess(self, x):
        return ' '.join([self.SOS_TOKEN, x, self.EOS_TOKEN])

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sentence = self.sentences[index]
        indices = [self.stoi[word] for word in sentence.split(' ')]

        return torch.LongTensor(indices)

