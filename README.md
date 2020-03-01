# CoLA Baselines

Baselines accompanying paper [Neural Network Acceptability Judgments](https://www.nyu.edu/projects/bowman/neural_network_acceptability.pdf). Check the [CoLA website](https://nyu-mll.github.io/CoLA) to download The Corpus of Linguistic Acceptability (CoLA), for more information, and for a demo model.

## Dataset

Training and validation sets for CoLA are available under [acceptability_corpus/raw](acceptability_corpus/raw) with a tokenized version available under [tokenized](acceptability_corpus/tokenized). Test data (unlabeled) is available here: [in domain](https://www.kaggle.com/c/cola-in-domain-open-evaluation) [out of domain](https://www.kaggle.com/c/cola-out-of-domain-open-evaluation). All models require tokenized data (we use the default NLTK tokenizer).

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

`python acceptability/run.py -d acceptability_corpus/tokenized`

This will use default classifier model and all of the default settings.

## Model

Our general model structure looks like figure below. Follow paper for more in-depth details.

![Model](https://i.imgur.com/eI4tNvd.png)

## Complex Run

Example of a command for running ELMo + Real/Fake on top of transferred encoder:

The directory containing data (value of `-d`) must contain three files, `train.tsv`, `dev.tsv` and `test.tsv`. Download vocabulary file used by us in our experiments from this [link](https://drive.google.com/file/d/14HNMByzrUM2ZJBjOqCzelFz5yJMHskFb/view?usp=sharing).

```
python acceptability/run.py -m linear_classifier -d data --save_loc save --vocab_file ./vocab_100k.tsv --logs_dir ./logs -g -r -p 40 -se 2 -n 1000 --encoder_path ./elmo_best_real_fake/experiment_lstm_pooling_elmo_h_528_l_3_lr_0.0001_e_360_do_0.2.pth --encoding_size 528 --embedding_size 217 --embedding_path ./elmo_best_real_fake/experiment_lstm_pooling_elmo_h_528_l_3_lr_0.0001_e_360_do_0.2.emb -lr 0.00005 -nl 3 -hs 1134 -do 0.2
```

## Pretrained Models and Testing

Pretrained models are available at this [link](https://drive.google.com/drive/folders/1HoHjdkc68fh7MTUBKAGZETGH5jfjsXR8?usp=sharing)

To do a test run over a dataset:

- Create your data-folder in same format as acceptability_corpus with `train.tsv`, `dev.tsv` and `test.tsv`.
- Download one of the pretrained encoders
- Download vocabulary file from this [link](https://drive.google.com/file/d/14HNMByzrUM2ZJBjOqCzelFz5yJMHskFb/view?usp=sharing)
- OPTIONAL: Train a classifier on CoLA using the command above
- Run (with the `-g` flag to use GPU hardware) the following command, e.g. for the downloadable ELMo real/fake encoder (without CoLA training) it would be something like:  
  `python acceptability/test.py -mf elmo.pth -vf vocab_100k.tsv -ef elmo.emb -d data/test.tsv -g`

To save the model predictions for each of the sentences in `test.tsv`, you can append to the last command the additional flag `-o predictions.txt`, which will write them in the `predictions.txt` file.

## Cite

If you use CoLA or the baselines in your research, please cite the accompanying paper using the following entry:

```
@article{warstadt2019neural,
  title={Neural network acceptability judgments},
  author={Warstadt, Alex and Singh, Amanpreet and Bowman, Samuel R},
  journal={Transactions of the Association for Computational Linguistics},
  volume={7},
  pages={625--641},
  year={2019},
  publisher={MIT Press}
}
```

## License

Baseline code is available under MIT license.

The text in this corpus is excerpted from the published works available on website, and copyright (where applicable) remains with the original authors or publishers. We expect that research use within the US is legal under fair use, but make no guarantee of this.
