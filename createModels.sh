#!/bin/bash

models=("/models/lstm_double.py")

for $i in "${models[@]}"; do
echo 'Model: ' $i
python trainmodel.py -model_location $i -embedding_location /embeddings/wordvec.bin -logs_dest /logs/wordvec1/
done
wait
