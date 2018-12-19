import torch
from torch.autograd import Variable
from acceptability.modules.dataset import AcceptabilityDataset, Vocab, GloVeIntersectedVocab
from acceptability.utils import seed_torch
from acceptability.utils import get_test_parser
from acceptability.modules.meter import Meter
from torch import nn


def test(args):
    vocab_path = args.vocab_file
    dataset_path = args.dataset_path
    gpu = args.gpu

    vocab = Vocab(vocab_path, True)
    dataset = AcceptabilityDataset(args, dataset_path, vocab)

    seed_torch(args)

    if gpu:
        model = torch.load(args.model_file)
    else:
        model = torch.load(args.model_file, map_location=lambda storage, loc: storage)

    if args.embedding_file is not None:
        if gpu:
            embedding = torch.load(args.embedding_file)
        else:
            embedding = torch.load(args.embedding_file, map_location=lambda storage, loc: storage)
    elif "glove" in args.embedding:
        vocab = GloVeIntersectedVocab(args, True)
        embedding = nn.Embedding(len(vocab.vectors), len(vocab.vectors[0]))
        embedding.weight.data.copy_(vocab.vectors)
        if gpu:
            embedding = embedding.cuda()


    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        pin_memory=gpu,
        shuffle=False
    )

    meter = Meter(2)

    model.eval()
    embedding.eval()
    outputs = []

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
        out_float = output.squeeze()
        output = (out_float > 0.5).long()
        # outputs.append(int(output))
        outputs.append(float(out_float))

        if not gpu:
            output = output.unsqueeze(0)

        meter.add(output.data, y.data)

    print("Matthews %.5f, Accuracy: %.5f" % (meter.matthews(), meter.accuracy()))
    if args.output_file != None:
        out_file = open(args.output_file, "w")
        for x in outputs:
            out_file.write(str(x) + "\n")
        out_file.close()


if __name__ == '__main__':
    parser = get_test_parser()
    test(parser.parse_args())
