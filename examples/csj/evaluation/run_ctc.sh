#!/bin/bash

# Select GPU
if [ $# -ne 2 ]; then
  echo "Error: set GPU number & config path." 1>&2
  echo "Usage: ./run_ctc.sh path_to_saved_model gpu_index" 1>&2
  exit 1
fi

# Set path to CUDA
export PATH=$PATH:/usr/local/cuda-8.0/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/extras/CUPTI/lib64

# Set path to python
PYTHON=/home/lab5/inaguma/.pyenv/versions/anaconda3-4.1.1/bin/python

model_path=$1
gpu_index=$2

epoch=-1
beam_width=100
eval_batch_size=1

CUDA_VISIBLE_DEVICES=$gpu_index $PYTHON eval_ctc.py \
  --model_path $model_path \
  --epoch $epoch \
  --beam_width $beam_width \
  --eval_batch_size $eval_batch_size
