from models import LSTMPoolingClassifier
from models import LinearClassifierWithEncoder
from models import CBOWClassifier


def get_model_instance(args):
    # Get embedding size from embedding parameter
    args.embedding_size = int(args.embedding.split('.')[-1][:-1])
    if args.model == "lstm_pooling_classifier":
        return LSTMPoolingClassifier(
            hidden_size=args.hidden_size,
            embedding_size=args.embedding_size,
            num_layers=args.num_layers
        )
    elif args.model == "aj_classifier":
        # TODO: Add support for encoder here later
        return LinearClassifierWithEncoder(
            hidden_size=args.hidden_size,
            embedding_size=args.embedding_size,
            encoding_size=args.encoding_size,
            num_layers=args.num_layers
        )
    elif args.model == "cbow_classifier":
        return CBOWClassifier(
            hidden_size=args.hidden_size,
            input_size=args.embedding_size,
            max_pool=args.max_pool
        )
    else:
        return None
