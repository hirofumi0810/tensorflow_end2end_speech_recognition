#!/bin/bash

RESULT_SAVE_PATH="/speech7/takashi01_nb/inaguma/models/tensorflow/librispeech/ctc/character/train100h/result_ensemble/blstm_ctc"
mkdir -p $RESULT_SAVE_PATH

# temp (train) == 1
# model1_path=/u/jp573469/inaguma/models/tensorflow/librispeech/ctc/character/train100h/blstm_ctc_320_5_rmsprop_lr1e-3_drop0.2_stack2_temp1
# model2_path=/u/jp573469/inaguma/models/tensorflow/librispeech/ctc/character/train100h/blstm_ctc_320_5_rmsprop_lr1e-3_drop0.2_stack2_temp1_2
# model3_path=/u/jp573469/inaguma/models/tensorflow/librispeech/ctc/character/train100h/blstm_ctc_320_5_rmsprop_lr1e-3_drop0.2_stack2_temp1_3
# model4_path=/u/jp573469/inaguma/models/tensorflow/librispeech/ctc/character/train100h/blstm_ctc_320_5_rmsprop_lr1e-3_drop0.2_stack2_temp1_7

# temp (train) == 2
model1_path=/dccstor/ichikaw01_nb/inaguma/blstm_ctc_320_5_rmsprop_lr1e-3_drop0.2_stack2_temp2
model2_path=/dccstor/ichikaw01_nb/inaguma/blstm_ctc_320_5_rmsprop_lr1e-3_drop0.2_stack2_temp2_1
model3_path=/dccstor/ichikaw01_nb/inaguma/blstm_ctc_320_5_rmsprop_lr1e-3_drop0.2_stack2_temp2_2
model4_path=/u/jp573469/inaguma/models/tensorflow/librispeech/ctc/character/train100h/blstm_ctc_320_5_rmsprop_lr1e-3_drop0.2_stack2_temp2_3

beam_width=100
temperature_infer=2

source activate tensorflow

python eval_ensemble4_ctc.py \
  --result_save_path $RESULT_SAVE_PATH \
  --model1_path $model1_path \
  --model2_path $model2_path \
  --model3_path $model3_path \
  --model4_path $model4_path \
  --epoch_model1 -1 \
  --epoch_model2 -1 \
  --epoch_model3 -1 \
  --epoch_model4 -1 \
  --beam_width $beam_width \
  --temperature_infer $temperature_infer
