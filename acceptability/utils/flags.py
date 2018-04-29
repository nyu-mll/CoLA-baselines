from __future__ import print_function
from datetime import datetime
import argparse
import torch

def get_parser():
    parser = argparse.ArgumentParser(description="Acceptability Judgments")
    parser.add_argument("-d", "--data", type=str, default="./data",
                        help="Directory containing train.txt, test.txt" +
                        "and valid.txt")
    parser.add_argument("-e", "--embedding", type=str, default="glove.6B.300d",
                        help="Embedding type to be used, select from" +
                        "http://torchtext.readthedocs.io/en/latest/vocab.html#pretrained-aliases")

    # Preprocess arguments
    parser.add_argument("--should_not_preprocess_data", action="store_true", default=False,
                        help="Whether to preprocess data? Default: true (Will preprocess)")
    parser.add_argument("--should_not_lowercase", action="store_true", default=False,
                        help="Should lowercase data? Default: true (Will lowercase)")
    parser.add_argument("--preprocess_tokenizer", default='space', type=str,
                        help="Type of tokenizer to use (space|nltk)")

    parser.add_argument("-v", "--vocab_file", type=str, default="./vocab_100k.tsv",
                        help="File containing vocabulary to be used with embedding")
    parser.add_argument("--glove", action="store_true", default=False,
                        help="Whether to use GloVE embedidngs for models")
    parser.add_argument("-es", "--embedding_size", type=int, default=300,
                        help="Embedding dimension for custom embedding")
    parser.add_argument("-ep", "--embedding_path", type=str, default=None,
                        help="If specified, custom embedding will be loaded from this path")
    parser.add_argument("--train_embeddings", action="store_true", default=False,
                        help="Whether to train embeddings?")
    parser.add_argument("--imbalance", action="store_true", default=False,
                        help="Is there a class imbalance?")

    parser.add_argument("-l", "--logs_dir", type=str, default="./logs",
                        help="Log directory")
    parser.add_argument("--should_not_log", action='store_true',
                        help="Specify when trainer should not log to file")

    parser.add_argument("-dt", "--data_type", type=str,
                        default="discriminator",
                        help="Data type")
    # TODO: Take a look on how to make this enum later
    parser.add_argument("-m", "--model", type=str,
                        default="lstm_pooling_classifier",
                        help="Type of the model you want to use")
    parser.add_argument("-s", "--save_loc", type=str, default="./save",
                        help="Save point for models")
    parser.add_argument("-g", "--gpu", action="store_true", default=False,
                        help="Whether use GPU or not")
    parser.add_argument("-cp", "--crop_pad_length", type=int, default=30,
                        help="Padding Crop length")

    parser.add_argument("--seed", type=int, default=1111,
                        help="Seed for reproducability")
    parser.add_argument("-bs", "--buffer_size", type=int, default=1,
                        help="Buffer size for logger")
    # Chunk parameters
    parser.add_argument("-se", "--stages_per_epoch", type=int, default=2,
                        help="Eval/Stats steps per epoch")
    parser.add_argument("--prints_per_stage", type=int, default=1,
                        help="How many times print stats per epoch")
    parser.add_argument("-p", "--patience", type=int, default=20,
                        help="Early stopping patience")
    parser.add_argument("-n", "--epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=32,
                        help="Batch size")

    # by_source doesn'tw ork at the moment
    parser.add_argument("--by_source", action="store_true", default=False,
                        help="Break output stats by source")
    parser.add_argument("--max_pool", action="store_true", default=False,
                        help="Use max pooling for CBOW")

    # Tuneable parameters
    parser.add_argument("-hs", "--hidden_size", type=int, default=300,
                        help="Hidden dimension for LSTM")
    parser.add_argument("-nl", "--num_layers", type=int, default=1,
                        help="Number of layers for LSTM")
    parser.add_argument("-lr", "--learning_rate", type=float, default=.0005,
                        help="Learning rate")
    parser.add_argument("-do", "--dropout", type=float, default=0.5,
                        help="Dropout")


    # Encoder parameter
    parser.add_argument("--encoding_size", type=int, default=100,
                        help="Output size of encoder, input size of linear")
    parser.add_argument("--encoder_num_layers", type=int, default=1,
                        help="Number of layers in encoder network")

    ## Take care to pass this argument for loading a pretrained encoder
    parser.add_argument("--encoder_path", type=str, default=None,
                        help="Location of encoder checkpoint")
    parser.add_argument("--encoding_type", type=str,
                        default="lstm_pooling_classifier",
                        help="Class of encoder")

    # Train dataset evaluate parameters
    # Can be useful when train dataset is small (like directly evaluating acceptability dataset)
    parser.add_argument("--evaluate_train", action="store_true", default=False,
                        help="Whether to evaluate training set after some interval (default: False)")
    parser.add_argument("--train_evaluate_interval", type=int, default=10,
                        help="Interval after which train dataset needs to be evaluated.")

    parser.add_argument("--experiment_name", type=str,
                        default=None,
                        help="Name of the current experiment")
    parser.add_argument("-rf", "--resume_file", type=str,
                        default=None,
                        help="Use specific checkpoint path for resuming")
    parser.add_argument("-r", "--resume", action="store_true", default=False,
                        help="Whether should resume training?" +
                        " Will look for checkpoint with experiment name")
    return parser


def get_lm_parser():
    parser = argparse.ArgumentParser("Acceptability Judgments LM")
    parser.add_argument("-d", "--data", type=str,
                        help="Directory containing train.tsv and valid.tsv")
    parser.add_argument("-v", "--vocab_file", type=str,
                        help="Vocabulary file")

    parser.add_argument("-m", "--model", type=str, default="lstm",
                        help="Model to be used for LM")
    parser.add_argument("-l", "--logs_dir", type=str, default="./logs",
                        help="Folder for storing logs")
    parser.add_argument("--should_not_log", action='store_true',
                        help="Specify when trainer should not log to file")
    parser.add_argument("-se", "--stages_per_epoch", type=int, default=2,
                        help="Eval/Stats steps per epoch")

    parser.add_argument("-p", "--patience", type=int, default=20,
                        help="Early stopping patience")

    parser.add_argument("--seed", type=int, default=1111,
                        help="Seed for reproducability")

    parser.add_argument("-en", "--experiment_name", type=str, default=None,
                        help="Name of the experiment")
    parser.add_argument("-es", "--embedding_size", type=int, default=300,
                        help="Size of the embedding dimension")
    parser.add_argument("-sl", "--seq_length", type=int, default=35,
                        help="Sequence length")
    parser.add_argument("-hs", "--hidden_size", type=int, default=600,
                        help="Size of the hidden dimension")
    parser.add_argument("-nl", "--num_layers", type=int, default=1,
                        help="Size of the hidden dimension")
    parser.add_argument("-b", "--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("-bs", "--buffer_size", type=int, default=1,
                        help="Buffer size for logger")
    parser.add_argument("-e", "--epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("-do", "--dropout", type=float, default=0.5,
                        help="Dropout")
    parser.add_argument("-g", "--gpu", action="store_true", default=torch.cuda.is_available(),
                        help="GPU")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("-s", "--save_loc", type=str, default='./save',
                        help="Save folder")
    parser.add_argument("-rf", "--resume_file", type=str,
                        default=None,
                        help="Use specific checkpoint path for resuming")
    parser.add_argument("-r", "--resume", action="store_true", default=False,
                        help="Whether should resume training?" +
                        " Will look for checkpoint with experiment name")
    parser.add_argument('--clip', type=float, default=0.5,
                        help='gradient clipping')
    return parser

def get_lm_generator_parser():
    parser = argparse.ArgumentParser(description='Acceptability LM Generator')

    # Model parameters.
    parser.add_argument("-m", "--checkpoint", type=str, default="./model.pth",
                        help="model checkpoint to use")
    parser.add_argument("-o", "--outf", type=str, default="generated.txt",
                        help="output file for generated text")
    parser.add_argument("-n", "--nlines", type=int, default="1000",
                        help="number of lines to generate")
    parser.add_argument("-v", "--vocab_file", type=str, default="vocab_100k.tsv",
                        help="number of lines to generate")
    parser.add_argument("--seed", type=int, default=1111,
                        help="random seed")
    # TODO: Change default value to False and check later explicity
    parser.add_argument("-g", "--gpu", action="store_true", default=torch.cuda.is_available(),
                        help="use CUDA")
    parser.add_argument("-t", "--temperature", type=float, default=1.0,
                        help="temperature - higher will increase diversity")
    parser.add_argument("--log_interval", type=int, default=100,
                        help="reporting interval")

    return parser

def get_lm_evaluator_parser():
    parser = argparse.ArgumentParser(description='Acceptability LM Evaluator')

    # Model parameters.
    parser.add_argument("-m", "--checkpoint", type=str, default="./model.pth",
                        help="model checkpoint to use")
    parser.add_argument("-d", "--data", type=str,
                        help="Directory containing data.tsv")
    parser.add_argument("-o", "--outf", type=str, default="generated.txt",
                        help="output file for log probs")
    parser.add_argument("-v", "--vocab_file", type=str, default="vocab_100k.tsv",
                        help="vocab location")
    parser.add_argument("--seed", type=int, default=1111,
                        help="random seed")
    # TODO: Change default value to False and check later explicity
    parser.add_argument("-g", "--gpu", action="store_true", default=torch.cuda.is_available(),
                        help="use CUDA")
    parser.add_argument("--log_interval", type=int, default=100,
                        help="reporting interval")
    parser.add_argument("-b", "--batch_size", type=int, default=32,
                        help="Batch size")

    return parser

# python -u acceptability/lm_evaluate.py -d acceptability_corpus/tokenized/in_domain_test.tsv -m checkpoints/experiment_lstm_s_35_h_891_l_2_lr_0.0002_d_0.20.pth -o logs -v ../data/vocabs/vocab_100k.tsv

def get_test_parser():
    parser = argparse.ArgumentParser(description='Acceptability Test')

    parser.add_argument("-mf", "--model_file", type=str, help="Model file to load")
    parser.add_argument("-vf", "--vocab_file", type=str, help="Vocab file to load")
    parser.add_argument("-ef", "--embedding_file", type=str, help="Embedding file to load")
    parser.add_argument("-d", "--dataset_path", type=str, help="Test file")
    parser.add_argument("-s", "--seed", type=int, default=11111, help="Random seed")
    parser.add_argument("-g", "--gpu", action="store_true", default=False, help="Use GPU")

    # Preprocess arguments
    parser.add_argument("--should_not_preprocess_data", action="store_true", default=False,
                        help="Whether to preprocess data? Default: true (Will preprocess)")
    parser.add_argument("--should_not_lowercase", action="store_true", default=False,
                        help="Should lowercase data? Default: true (Will lowercase)")
    parser.add_argument("--preprocess_tokenizer", default='space', type=str,
                        help="Type of tokenizer to use (space|nltk)")
    parser.add_argument("-cp", "--crop_pad_length", type=int, default=30,
                        help="Padding Crop length")


    return parser
