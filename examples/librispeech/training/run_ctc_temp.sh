#!/bin/bash

MODEL_SAVE_PATH="/n/sd8/inaguma/result/tensorflow/librispeech"

# Select GPU
if [ $# -lt 2 ]; then
  echo "Error: set GPU number & config path." 1>&2
  echo "Usage: ./run_ctc_temp.sh path_to_config_file gpu_index1 gpu_index2... (arbitrary number)" 1>&2
  exit 1
fi

# Set path to CUDA
export PATH=$PATH:/usr/local/cuda-8.0/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/extras/CUPTI/lib64

# Set path to python
PYTHON=/home/lab5/inaguma/.pyenv/versions/anaconda3-4.1.1/bin/python

# export PATH=$PATH:/opt/share/cuda-8.0/x86_64/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/x86_64/lib64:/usr/local/cuda-8.0/x86_64/extras/CUPTI/lib64:/opt/share/cuDNN-v5.1-8.0/cuda/lib64
# PYTHON=/u/jp573469/.pyenv/shims/python

gpu_num=`expr $# - 1`
config_path=$1
gpu_index=$2
filename=$(basename $config_path | awk -F. '{print $1}')

if [ $# -ne 2 ]; then
  rest_gpu_num=`expr $gpu_num - 1`
  for i in `seq 1 $rest_gpu_num`
  do
    gpu_index=$gpu_index","${3}
    shift
  done
fi

mkdir -p log

# Background job version
CUDA_VISIBLE_DEVICES=$gpu_index nohup $PYTHON train_ctc_temp.py \
  $config_path $MODEL_SAVE_PATH $gpu_index > log/$filename".log" &

# Standard output version
# CUDA_VISIBLE_DEVICES=$gpu_index $PYTHON train_ctc.py \
#   $config_path $MODEL_SAVE_PATH $gpu_index
