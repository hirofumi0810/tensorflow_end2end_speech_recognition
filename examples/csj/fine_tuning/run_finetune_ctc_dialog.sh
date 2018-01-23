#!/bin/zsh

# select GPU
if [ $# -ne 3 ]; then
  echo "Error: set GPU number & config path." 1>&2
  echo "Usage: ./ctc.sh gpu_num path_to_config_file path_to_trained_model." 1>&2
  exit 1
fi

# GPU setting
export PATH=$PATH:/usr/local/cuda-8.0/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64
PYTHON=/home/lab5/inaguma/.pyenv/versions/anaconda3-4.1.1/bin/python

filename=$(basename $2 | awk -F. '{print $1}')
CUDA_VISIBLE_DEVICES=$1 nohup $PYTHON finetune_ctc_dialog.py $2 $3 > log/$filename".log" &
# CUDA_VISIBLE_DEVICES=$1 $PYTHON finetune_ctc_dialog.py $2 $3
