#!/bin/bash

#python trainModel.py models/lstm_double.py logs/lstm1win2_hid30/ embeddings/PubMed-shuffle-win-2.bin --value 30
#python evaluateModel.py models/lstm_double.py logs/lstm1win2_hid30/ embeddings/PubMed-shuffle-win-2.bin --value 30

#python trainModel.py models/lstm_double.py logs/lstm1win30_hid30/ embeddings/PubMed-shuffle-win-30.bin --value 30
#python evaluateModel.py models/lstm_double.py logs/lstm1win30_hid30/ embeddings/PubMed-shuffle-win-30.bin --value 30

#python trainModel.py models/lstm_single_embedFixed.py logs/lstm2iter17_fixed/ embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin
#python evaluateModel.py models/lstm_single.py logs/lstm2iter17_fixed/ embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin

#python trainModel.py models/lstm_single_embedDual.py logs/lstm2iter17_dual/ embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin
#python evaluateModel.py models/lstm_single_embedDual.py logs/lstm2iter17_dual/ embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin

hidden=("20" "300" "400")

#for v in "${hidden[@]}"
#do
#  echo models/lstm_single.py logs/lstm2iter17_hid$v/ embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin --value $v
#  python trainModel.py models/lstm_single.py logs/lstm2iter17_hid$v/ embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin --value $v
#  python evaluateModel.py models/lstm_single.py logs/lstm2iter17_hid$v/ embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin --value $v
#done

#for v in "${hidden[@]}"
#do
#  echo models/lstm_double.py logs/lstm1iter17_hid$v/ embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin --value $v
#  python trainModel.py models/lstm_double.py logs/lstm1iter17_hid$v/ embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin --value $v
#  python evaluateModel.py models/lstm_double.py logs/lstm1iter17_hid$v/ embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin --value $v
#done

for v in "${hidden[@]}"
do
echo _______________ $v _______________
#python trainModel.py models/bilstm_single.py logs/bilstm2iter17_drop_hid$v/ embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin --value $v
#python evaluateModel.py models/bilstm_single.py logs/bilstm2iter17_drop_hid$v/ embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin --value $v
#python evaluateModelF.py models/bilstm_single.py logs/bilstm2iter17_drop_hid$v/ embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin --value $v

python trainModel.py models/lstm_single.py logs/lstm2iter17_hid$v/ embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin --value $v
python evaluateModel.py models/lstm_single.py logs/lstm2iter17_hid$v/ embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin --value $v
python evaluateModelF.py models/lstm_single.py logs/lstm2iter17_hid$v/ embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin --value $v

python trainModel.py models/lstm_single_embedDual.py logs/lstm2iter17_dual_hid$v/ embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin --value $v
python evaluateModel.py models/lstm_single_embedDual.py logs/lstm2iter17_dual_hid$v/ embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin --value $v
python evaluateModelF.py models/lstm_single_embedDual.py logs/lstm2iter17_dual_hid$v/ embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin --value $v

python trainModel.py models/lstm_single_embedFixed.py logs/lstm2iter17_fix_hid100/ embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin --value 100
python evaluateModel.py models/lstm_single_embedFixed.py logs/lstm2iter17_fix_hid100/ embeddings/low_shuff_combine_tokenized.txt-iter17-min5.bin --value 100
done
