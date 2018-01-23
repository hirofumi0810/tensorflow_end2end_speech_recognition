#!/bin/zsh

# select GPU
if [ $# -ne 1 ]; then
  echo "Error: set GPU number." 1>&2
  exit 1
fi

# GPU setting
export PATH=$PATH:/usr/local/cuda-8.0/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64
PYTHON=/home/lab5/inaguma/.pyenv/versions/anaconda3-4.1.1/bin/python

# phone1
# is13:
# fbank:

# phone2
# is13:
# fbank:
# joint:

# phone41
# is13:
# fbank:
# joint:

# config
label_type=phone41 # phone1, phone2, phone41
feature=fbank  # fbank, is13, joint
model=blstm
layer=5
cell=256
optimizer=rmsprop
lr=1e-3
drop_in=0.8
drop_h=0.5

CUDA_VISIBLE_DEVICES=$1 nohup $PYTHON eval_ctc.py --label_type $label_type --feature $feature --model $model --layer $layer --cell $cell --optimizer $optimizer --lr $lr --drop_in $drop_in --drop_h $drop_h > log/$model"_"ctc"_"$optimizer"_"$label_type"_"$feature"_"$cell"_"$layer".log" &
