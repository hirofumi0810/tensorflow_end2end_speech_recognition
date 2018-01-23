#!/bin/bash

# Select GPU
if [ $# -ne 2 ]; then
  echo "Error: set GPU number & config path." 1>&2
  echo "Usage: ./run_ctc.sh path_to_saved_model gpu_index" 1>&2
  exit 1
fi

# Set path to CUDA
# export PATH=$PATH:/usr/local/cuda-8.0/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/extras/CUPTI/lib64

# Set path to python
# PYTHON=/home/lab5/inaguma/.pyenv/versions/anaconda3-4.1.1/bin/python

export PATH=$PATH:/opt/share/cuda-8.0/x86_64/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/x86_64/lib64:/usr/local/cuda-8.0/x86_64/extras/CUPTI/lib64:/opt/share/cuDNN-v5.1-8.0/cuda/lib64
PYTHON=/u/jp573469/.pyenv/shims/python

model_path=$1
gpu_index=$2

epoch=-1
eval_batch_size=-1
temperature=1

CUDA_VISIBLE_DEVICES=$gpu_index $PYTHON save_ctc_prob.py \
  --model_path $model_path \
  --epoch $epoch \
  --eval_batch_size $eval_batch_size \
  --temperature $temperature
