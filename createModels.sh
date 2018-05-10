#!/bin/bash

#python trainModel.py models/lstm_double.py logs/lstm1win2/ embeddings/PubMed-shuffle-win-2.bin

#python trainModel.py models/lstm_single_embedFixed.py logs/lstm2iter17_fixed/ embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin
#python evaluateModel.py models/lstm_single.py logs/lstm2iter17_fixed/ embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin

#python trainModel.py models/lstm_single_embedDual.py logs/lstm2iter17_dual/ embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin
#python evaluateModel.py models/lstm_single_embedDual.py logs/lstm2iter17_dual/ embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin

hidden=("40" "50" "100" "150" "200" "500")

#for v in "${hidden[@]}"
#do
#  echo models/lstm_single.py logs/lstm2iter17_hid$v/ embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin --value $v
#  python trainModel.py models/lstm_single.py logs/lstm2iter17_hid$v/ embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin --value $v
#  python evaluateModel.py models/lstm_single.py logs/lstm2iter17_hid$v/ embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin --value $v
#done

for v in "${hidden[@]}"
do
  echo models/lstm_double.py logs/lstm1iter17_hid$v/ embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin --value $v
  python trainModel.py models/lstm_double.py logs/lstm1iter17_hid$v/ embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin --value $v
  python evaluateModel.py models/lstm_double.py logs/lstm1iter17_hid$v/ embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin --value $v
done
