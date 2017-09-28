#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate the trained CTC model (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf
import yaml
import argparse

sys.path.append(os.path.abspath('../../../'))
from experiments.librispeech.data.load_dataset_ctc import Dataset
from experiments.librispeech.metrics.ctc import do_eval_cer, do_eval_wer
from models.ctc.vanilla_ctc import CTC

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
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
    test_clean_data = Dataset(
        data_type='dev_clean', train_data_size=params['train_data_size'],
        label_type=params['label_type'],
        batch_size=1, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=False)
    test_other_data = Dataset(
        data_type='dev_other', train_data_size=params['train_data_size'],
        label_type=params['label_type'],
        batch_size=1, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=False)

    with tf.name_scope('tower_gpu0'):
        # Define placeholders
        model.create_placeholders()

        # Add to the graph each operation (including model definition)
        _, logits = model.compute_loss(model.inputs_pl_list[0],
                                       model.labels_pl_list[0],
                                       model.inputs_seq_len_pl_list[0],
                                       model.keep_prob_input_pl_list[0],
                                       model.keep_prob_hidden_pl_list[0],
                                       model.keep_prob_output_pl_list[0])
        decode_op = model.decoder(logits,
                                  model.inputs_seq_len_pl_list[0],
                                  decode_type='beam_search',
                                  beam_width=20)

    # Create a saver for writing training checkpoints
    saver = tf.train.Saver()

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(model.save_path)

        # If check point exists
        if ckpt:
            # Use last saved model
            model_path = ckpt.model_checkpoint_path
            if epoch != -1:
                model_path = model_path.split('/')[:-1]
                model_path = '/'.join(model_path) + '/model.ckpt-' + str(epoch)
            saver.restore(sess, model_path)
            print("Model restored: " + model_path)
        else:
            raise ValueError('There are not any checkpoints.')

        print('Test Data Evaluation:')
        if 'char' in params['label_type']:
            cer_clean_test, wer_clean_test = do_eval_cer(
                session=sess,
                decode_ops=[decode_op],
                model=model,
                dataset=test_clean_data,
                label_type=params['label_type'],
                eval_batch_size=params['batch_size'],
                progressbar=True)
            print('  CER (clean): %f %%' % (cer_clean_test * 100))
            print('  WER (clean): %f %%' % (wer_clean_test * 100))

            cer_other_test, wer_other_test = do_eval_cer(
                session=sess,
                decode_ops=[decode_op],
                model=model,
                dataset=test_other_data,
                label_type=params['label_type'],
                eval_batch_size=params['batch_size'],
                progressbar=True)
            print('  CER (other): %f %%' % (cer_other_test * 100))
            print('  WER (other): %f %%' % (wer_other_test * 100))
        else:
            wer_clean_test = do_eval_wer(
                session=sess,
                decode_ops=[decode_op],
                model=model,
                dataset=test_clean_data,
                train_data_size=params['train_data_size'],
                is_test=True,
                eval_batch_size=params['batch_size'],
                progressbar=True)
            print('  WER (clean): %f %%' % (wer_clean_test * 100))

            wer_other_test = do_eval_wer(
                session=sess,
                decode_ops=[decode_op],
                model=model,
                dataset=test_other_data,
                train_data_size=params['train_data_size'],
                is_test=True,
                eval_batch_size=params['batch_size'],
                progressbar=True)
            print('  WER (other): %f %%' % (wer_other_test * 100))


def main():

    args = parser.parse_args()

    # Load config file
    with open(os.path.join(args.model_path, 'config.yml'), "r") as f:
        config = yaml.load(f)
        params = config['param']

    # Except for a blank class
    if params['label_type'] == 'character':
        params['num_classes'] = 28
    elif params['label_type'] == 'character_capital_divide':
        params['num_classes'] = 77
    elif params['label_type'] == 'word':
        if params['train_data_size'] == 'train_clean100':
            params['num_classes'] = 7213
        elif params['train_data_size'] == 'train_clean360':
            params['num_classes'] = 16287
        elif params['train_data_size'] == 'train_other500':
            params['num_classes'] = 18669
        elif params['train_data_size'] == 'train_all':
            params['num_classes'] = 26642

    # Model setting
    model = CTC(
        encoder_type=params['encoder_type'],
        input_size=params['input_size'] * params['num_stack'],
        num_units=params['num_units'],
        num_layers=params['num_layers'],
        num_classes=params['num_classes'],
        parameter_init=params['weight_init'],
        clip_grad=params['clip_grad'],
        clip_activation=params['clip_activation'],
        num_proj=params['num_proj'],
        weight_decay=params['weight_decay'])

    model.save_path = args.model_path
    do_eval(model=model, params=params, epoch=args.epoch)


if __name__ == '__main__':
    main()
