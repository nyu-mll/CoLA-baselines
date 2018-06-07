# CoLA Baselines

Baselines accompanying paper [Neural Network Acceptability Judgments](https://www.nyu.edu/projects/bowman/neural_network_acceptability.pdf). Check the [CoLA website](https://nyu-mll.github.io/CoLA) to download The Corpus of Linguistic Acceptablity (CoLA), for more information, and for a demo model.

## Dataset

Training and validation sets for CoLA are available under [acceptability_corpus/raw](acceptability_corpus/raw) with a tokenized version available under [tokenized](acceptability_corpus/tokenized).

## Requirements

- Python >= 3
- PyTorch v0.3.0
- TorchNet and TorchText
- NLTK (Optional: For NLTK preprocessing)

## Running

Install [Pytorch](https://pytorch.org/) v0.3.0.

Then run:

```
git clone https://github.com/nyu-mll/acceptability-judgments.git
cd acceptability-judgments
pip3 install -r requirements.txt
python setup.py develop
```

Run a simple training session by:

`python acceptability/run.py -d acceptability_corpus/raw`

This will use default classifier model and all of the default settings.

## Model

Our general model structure looks like figure below. Follow paper for more in-depth details.

![Model](https://i.imgur.com/eI4tNvd.png)

## Complex Run

Example of a command for running ELMo + Real/Fake on top of transferred encoder:

Folder containing data must contain three files, `train.tsv`, `dev.tsv` and `test.tsv`. Download vocabulary file used by us in our experiments from this [link](https://drive.google.com/file/d/14HNMByzrUM2ZJBjOqCzelFz5yJMHskFb/view?usp=sharing).

```
python acceptability/run.py -m linear_classifier -d data --save_loc save --vocab_file ./vocab_100k.tsv --logs_dir ./logs -g -r -p 40 -se 2 -n 1000 --encoder_path ./elmo_best_real_fake/experiment_lstm_pooling_elmo_h_528_l_3_lr_0.0001_e_360_do_0.2.pth --encoding_size 528 --embedding_size 217 --embedding_path ./elmo_best_real_fake/experiment_lstm_pooling_elmo_h_528_l_3_lr_0.0001_e_360_do_0.2.emb -lr 0.00005 -nl 3 -hs 1134 -do 0.2
```

## Pretrained Models and Testing

Pretrained models are available at this [link](https://drive.google.com/drive/folders/1HoHjdkc68fh7MTUBKAGZETGH5jfjsXR8?usp=sharing)

To do a test run over a dataset:

- Create your data-folder in same format as acceptability_corpus with `train.tsv`, `dev.tsv` and `test.tsv`.
- Download one of the pretrained models
- Download vocabulary file from this [link](https://drive.google.com/file/d/14HNMByzrUM2ZJBjOqCzelFz5yJMHskFb/view?usp=sharing)
- Run the following command, for e.g ELMo pretrained it would be something like:
- `python acceptability/test.py -mf elmo.pth -vf vocab_100k.tsv -ef elmo.emb -d data/test.tsv -g`

## Cite

Cite CoLA or the baselines using the following entry:

```
@article{warstadt2018neural,
  title={Neural Network Acceptability Judgments},
  author={Warstadt, Alex and Singh, Amanpreet and Bowman, Samuel R},
  journal={arXiv preprint arXiv:1805.12471},
  year={2018}
}
```

## License

Baseline code is available under MIT license.

The text in this corpus is excerpted from the published works available on website, and copyright (where applicable) remains with the original authors or publishers. We expect that research use within the US is legal under fair use, but make no guarantee of this.
