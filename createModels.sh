#!/bin/bash

#python trainModel.py models/lstm_double.py logs/lstm1iter17/ embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin

#python trainModel.py models/lstm_double.py logs/lstm1win2/ embeddings/PubMed-shuffle-win-2.bin

python trainModel.py models/lstm_single_embedFixed.py logs/lstm2iter17/ embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin
python evaluateModel.py models/lstm_single.py logs/lstm2iter17/ embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin
