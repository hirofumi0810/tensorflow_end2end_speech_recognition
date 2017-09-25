#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate the trained Attention-based model (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf
import yaml
import argparse

sys.path.append(os.path.abspath('../../../'))
from experiments.timit.data.load_dataset_attention import Dataset
from experiments.timit.metrics.attention import do_eval_per, do_eval_cer
from models.attention import blstm_attention_seq2seq

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=-
                    1, help='the epoch to restore')
parser.add_argument('--model_path', type=str,
                    help='path to the model to evaluate')


def do_eval(model, params, epoch=None):
    """Evaluate the model.
    Args:
        model: the model to restore
        params (dict): A dictionary of parameters
        epoch (int): the epoch to restore
    """
    # Load dataset
    if 'phone' in params['label_type']:
        test_data = Dataset(
            data_type='test', label_type='phone39',
            batch_size=1, eos_index=params['eos_index'],
            splice=params['splice'],
            num_stack=params['num_stack'], num_skip=params['num_skip'],
            sort_utt=False, progressbar=True)
    else:
        test_data = Dataset(
            data_type='test', label_type=params['label_type'],
            batch_size=1, eos_index=params['eos_index'],
            splice=params['splice'],
            num_stack=params['num_stack'], num_skip=params['num_skip'],
            sort_utt=False, progressbar=True)
    # TODO(hirofumi): add frame_stacking and splice

    # Define placeholders
    model.create_placeholders()

    # Add to the graph each operation (including model definition)
    _, _, decoder_outputs_train, decoder_outputs_infer = model.compute_loss(
        model.inputs_pl_list[0],
        model.labels_pl_list[0],
        model.inputs_seq_len_pl_list[0],
        model.labels_seq_len_pl_list[0],
        model.keep_prob_input_pl_list[0],
        model.keep_prob_hidden_pl_list[0],
        model.keep_prob_output_pl_list[0])
    _, decode_op_infer = model.decoder(
        decoder_outputs_train,
        decoder_outputs_infer)
    per_op = model.compute_ler(
        model.labels_st_true_pl, model.labels_st_pred_pl)

    # Create a saver for writing training checkpoints
    saver = tf.train.Saver()

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(model.save_path)

        # If check point exists
        if ckpt:
            # Use last saved model
            model_path = ckpt.model_checkpoint_path
            if epoch != -1:
                # Use the best model
                # NOTE: In the training stage, parameters are saved only when
                # accuracies are improved
                model_path = model_path.split('/')[:-1]
                model_path = '/'.join(model_path) + '/model.ckpt-' + str(epoch)
            saver.restore(sess, model_path)
            print("Model restored: " + model_path)
        else:
            raise ValueError('There are not any checkpoints.')

        print('Test Data Evaluation:')
        if params['label_type'] in ['character', 'character_capital_divide']:
            cer_test, wer_test = do_eval_cer(
                session=sess,
                decode_op=decode_op_infer,
                model=model,
                dataset=test_data,
                label_type=params['label_type'],
                eval_batch_size=1,
                progressbar=True)
            print('  CER: %f %%' % (cer_test * 100))
            print('  WER: %f %%' % (wer_test * 100))
        else:
            per_test = do_eval_per(
                session=sess,
                decode_op=decode_op_infer,
                per_op=per_op,
                model=model,
                dataset=test_data,
                label_type=params['label_type'],
                eos_index=params['eos_index'],
                eval_batch_size=1,
                progressbar=True)
            print('  PER: %f %%' % (per_test * 100))


def main():

    args = parser.parse_args()

    # Load config file
    with open(os.path.join(args.model_path, 'config.yml'), "r") as f:
        config = yaml.load(f)
        params = config['param']

    params['sos_index'] = 0
    params['eos_index'] = 1
    if params['label_type'] == 'phone61':
        params['num_classes'] = 63
    elif params['label_type'] == 'phone48':
        params['num_classes'] = 50
    elif params['label_type'] == 'phone39':
        params['num_classes'] = 41
    elif params['label_type'] == 'character':
        params['num_classes'] = 30
    elif params['label_type'] == 'character_capital_divide':
        params['num_classes'] = 74

    # Model setting
    # AttentionModel = load(model_type=params['model'])
    model = blstm_attention_seq2seq.BLSTMAttetion(
        input_size=params['input_size'],
        encoder_num_unit=params['encoder_num_unit'],
        encoder_num_layer=params['encoder_num_layer'],
        attention_dim=params['attention_dim'],
        attention_type=params['attention_type'],
        decoder_num_unit=params['decoder_num_unit'],
        decoder_num_layer=params['decoder_num_layer'],
        embedding_dim=params['embedding_dim'],
        num_classes=params['num_classes'],
        sos_index=params['sos_index'],
        eos_index=params['eos_index'],
        max_decode_length=params['max_decode_length'],
        attention_smoothing=params['attention_smoothing'],
        attention_weights_tempareture=params['attention_weights_tempareture'],
        logits_tempareture=params['logits_tempareture'],
        parameter_init=params['weight_init'],
        clip_grad=params['clip_grad'],
        clip_activation_encoder=params['clip_activation_encoder'],
        clip_activation_decoder=params['clip_activation_decoder'],
        weight_decay=params['weight_decay'],
        beam_width=20)

    model.save_path = args.model_path
    do_eval(model=model, params=params, epoch=args.epoch)


if __name__ == '__main__':
    main()
