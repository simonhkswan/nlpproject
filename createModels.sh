#!/bin/bash

#python trainModel.py models/lstm_double.py logs/lstm1iter17/ embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin

#python trainModel.py models/lstm_double.py logs/lstm1win2/ embeddings/PubMed-shuffle-win-2.bin

#python trainModel.py models/lstm_single_embedFixed.py logs/lstm2iter17_fixed/ embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin
#python evaluateModel.py models/lstm_single.py logs/lstm2iter17_fixed/ embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin

#python trainModel.py models/lstm_single_embedDual.py logs/lstm2iter17_dual/ embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin
#python evaluateModel.py models/lstm_single_embedDual.py logs/lstm2iter17_dual/ embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin
hidden=("10" "20" "30" "40" "50" "60" "70" "80")

for v in "${hidden[@]}"
do
  python trainModel.py models/lstm_single.py logs/lstm2iter17_hid$v/ embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin -i $v
done
