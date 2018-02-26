# acceptability

The acceptability judgments corpus can be found in acceptability_corpus/corpus_table.

To launch a training session, run training.run_training.py with flags as command line arguments, e.g:
python -m training.run_training.py --embedding_size 300 --data_dir /scratch/asw462/data/discriminator/ --data_type discriminator --vocab_path /scratch/asw462/data/bnc-30/vocab_20000.txt --log_path /scratch/asw462/logs/ --convergence_threshold 20 --num_layers 3 --crop_pad_length 30 --ckpt_path /scratch/asw462/models/ --batch_size 32 --prints_per_stage 1 --max_epochs 100 --embedding_path /scratch/asw462/data/bnc-30/embeddings_20000.txt --stages_per_epoch 100 --model_type rnn_classifier_pooling --gpu  --hidden_size 1383 --learning_rate 0.00251670745856 --experiment_name sweep_0115223201_rnn_classifier_pooling_19-lr0.0025-h_size1383-datadiscriminator-num_layers3"