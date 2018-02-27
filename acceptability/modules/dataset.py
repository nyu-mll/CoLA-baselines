import os
import nltk
import torch
from torch.utils.data import Dataset
from torchtext import vocab, data


class AcceptabilityDataset(Dataset):
    def __init__(self, path, vocab_name):
        self.pairs = []
        self.sentences = []
        if not os.exists(path):
            # TODO: log failure here
            os.exit(1)

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


def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def preprocess_label(label):
    if float(label) > 0:
        return '1'
    else:
        return '0'

def get_datasets(args):
    sentence = data.Field(
        sequential=True,
        fix_length=args.crop_pad_length,
        tokenize=tokenize,
        tensor_type=torch.cuda.LongTensor if args.gpu else torch.LongTensor,
        lower=True,
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
