#!/bin/bash

RESULT_SAVE_PATH="/speech7/takashi01_nb/inaguma/models/tensorflow/librispeech/ctc/character/train100h/result_ensemble/blstm_ctc"
mkdir -p $RESULT_SAVE_PATH

# temp (train) == 1
model1_path=/u/jp573469/inaguma/models/tensorflow/librispeech/ctc/character/train100h/blstm_ctc_320_5_rmsprop_lr1e-3_drop0.2_stack2_temp1
model2_path=/dccstor/ichikaw01_nb/inaguma/blstm_ctc_320_5_rmsprop_lr1e-3_drop0.2_stack2_temp1_1
model3_path=/u/jp573469/inaguma/models/tensorflow/librispeech/ctc/character/train100h/blstm_ctc_320_5_rmsprop_lr1e-3_drop0.2_stack2_temp1_2
model4_path=/u/jp573469/inaguma/models/tensorflow/librispeech/ctc/character/train100h/blstm_ctc_320_5_rmsprop_lr1e-3_drop0.2_stack2_temp1_3
model5_path=/dccstor/ichikaw01_nb/inaguma/blstm_ctc_320_5_rmsprop_lr1e-3_drop0.2_stack2_temp1_4
model6_path=/dccstor/ichikaw01_nb/inaguma/blstm_ctc_320_5_rmsprop_lr1e-3_drop0.2_stack2_temp1_5
model7_path=/dccstor/ichikaw01_nb/inaguma/blstm_ctc_320_5_rmsprop_lr1e-3_drop0.2_stack2_temp1_6
model8_path=/u/jp573469/inaguma/models/tensorflow/librispeech/ctc/character/train100h/blstm_ctc_320_5_rmsprop_lr1e-3_drop0.2_stack2_temp1_7


beam_width=100
temperature_infer=2

source activate tensorflow

python eval_ensemble8_ctc.py \
  --result_save_path $RESULT_SAVE_PATH \
  --model1_path $model1_path \
  --model2_path $model2_path \
  --model3_path $model3_path \
  --model4_path $model4_path \
  --model5_path $model5_path \
  --model6_path $model6_path \
  --model7_path $model7_path \
  --model8_path $model8_path \
  --epoch_model1 -1 \
  --epoch_model2 -1 \
  --epoch_model3 -1 \
  --epoch_model4 -1 \
  --epoch_model5 -1 \
  --epoch_model6 -1 \
  --epoch_model7 -1 \
  --epoch_model8 -1 \
  --beam_width $beam_width \
  --temperature_infer $temperature_infer
