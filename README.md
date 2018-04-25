# Acceptability Judgements

Dataset for testing grammatical acceptability of a sentence.

## Dataset

Dataset for grammaticality judgments is available under [acceptability_corpus/raw](acceptability_corpus/raw)

## Running

Install [PyTorch](https://pytorch.org/). Then run:

```
git clone https://github.com/nyu-mll/acceptability-judgments.git
cd acceptability-judgments
pip3 install -r requirements.txt
python setup.py develop
```

Run a simple training session by:

`python acceptability/run.py -d acceptability_corpus/raw`

This will use default classifier model and all of the default settings.

## Complex Run

Example of a complex run:

TODO: Fix this.

```
python acceptability/run.py --embedding_size 300 --data_dir /scratch/asw462/data/discriminator/ --data_type discriminator --vocab_path /scratch/asw462/data/bnc-30/vocab_20000.txt --log_path /scratch/asw462/logs/ --convergence_threshold 20 --num_layers 3 --crop_pad_length 30 --ckpt_path /scratch/asw462/models/ --batch_size 32 --prints_per_stage 1 --max_epochs 100 --embedding_path /scratch/asw462/data/bnc-30/embeddings_20000.txt --stages_per_epoch 100 --model_type rnn_classifier_pooling --gpu  --hidden_size 1383 --learning_rate 0.00251670745856 --experiment_name sweep_0115223201_rnn_classifier_pooling_19-lr0.0025-h_size1383-datadiscriminator-num_layers3"
```

## License
BSD