import torch

from torch import nn
from acceptability.models import LSTMPoolingClassifier
from acceptability.models import LinearClassifierWithEncoder
from acceptability.models import CBOWClassifier


def get_model_instance(args):
    # Get embedding size from embedding parameter
    args.embedding_size = int(args.embedding.split('.')[-1][:-1])
    if args.model == "lstm_pooling_classifier":
        return LSTMPoolingClassifier(
            hidden_size=args.hidden_size,
            embedding_size=args.embedding_size,
            num_layers=args.num_layers
        )
    elif args.model == "linear_classifier":
        # TODO: Add support for encoder here later
        return LinearClassifierWithEncoder(
            hidden_size=args.hidden_size,
            embedding_size=args.embedding_size,
            encoding_size=args.encoding_size,
            num_layers=args.num_layers,
            encoder_type=args.encoder_type,
            encoder_num_layers=args.encoder_num_layers,
            encoder_path=args.encoder_path
        )
    elif args.model == "cbow_classifier":
        return CBOWClassifier(
            hidden_size=args.hidden_size,
            input_size=args.embedding_size,
            max_pool=args.max_pool
        )
    else:
        return None

def get_encoder_instance(encoder_type, encoding_size, embedding_size,
                         encoder_num_layers,
                         encoder_path=None):

    encoder = lambda x: x
    if encoder_type == "lstm_pooling_classifier":
        encoder = LSTMPoolingClassifier(
            hidden_size=encoding_size,
            embedding_size=embedding_size,
            num_layers=encoder_num_layers
        )

        if encoder_path is not None:
            encoder.load_state_dict(torch.load(encoder_path))

            # Since we have loaded freeze params
            for p in encoder.parameters():
                p.requires_grad = False

    return encoder
