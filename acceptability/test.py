import torch
from torch.autograd import Variable
from acceptability.modules.dataset import AcceptabilityDataset, Vocab
from acceptability.utils import seed_torch
from acceptability.utils import get_test_parser


def test(args):
    vocab_path = args.vocab_file
    dataset_path = args.dataset_path
    gpu = args.gpu

    vocab = Vocab(vocab_path, True)
    dataset = AcceptabilityDataset(args, dataset_path, vocab)

    seed_torch(args)

    if gpu:
        model = torch.load(args.model_file)
        embedding = torch.load(args.embedding_file)
    else:
        model = torch.load(args.model_file, map_location=lambda storage, loc: storage)
        embedding = torch.load(args.embedding_file, map_location=lambda storage, loc: storage)


    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        pin_memory=gpu
    )

    model.eval()
    embedding.eval()

    for data in loader:
        x, y, _ = data
        x, y = Variable(x).long(), Variable(y)

        if gpu:
            x = x.cuda()
            y = y.cuda()

        x = embedding(x)

        output = model(x)

        if type(output) == tuple:
            output = output[0]
        output = output.squeeze()
        output = (output > 0.5).long()
        print(output,data[0])


if __name__ == '__main__':
    parser = get_test_parser()
    test(parser.parse_args())
