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

sys.path.append('../../../')
from experiments.librispeech.data.load_dataset_ctc import Dataset
from experiments.librispeech.metrics.ctc import do_eval_cer, do_eval_wer
from models.ctc.load_model import load


def do_eval(network, params, epoch=None):
    """Evaluate the model.
    Args:
        network: model to restore
        params: A dictionary of parameters
        epoch: int, the epoch to restore
    """
    # Load dataset
    test_clean_data = Dataset(
        data_type='test_clean',
        train_data_size=params['train_data_size'],
        label_type=params['label_type'], batch_size=params['batch_size'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=False)
    test_other_data = Dataset(
        data_type='test_other',
        train_data_size=params['train_data_size'],
        label_type=params['label_type'], batch_size=params['batch_size'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=False)

    # Define placeholders
    network.create_placeholders(gpu_index=None)

    # Add to the graph each operation (including model definition)
    _, logits = network.compute_loss(network.inputs_pl_list[0],
                                     network.labels_pl_list[0],
                                     network.inputs_seq_len_pl_list[0],
                                     network.keep_prob_input_pl_list[0],
                                     network.keep_prob_hidden_pl_list[0],
                                     network.keep_prob_output_pl_list[0])
    decode_op = network.decoder(logits,
                                network.inputs_seq_len_pl_list[0],
                                decode_type='beam_search',
                                beam_width=20)

    # Create a saver for writing training checkpoints
    saver = tf.train.Saver()

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(network.model_dir)

        # If check point exists
        if ckpt:
            # Use last saved model
            model_path = ckpt.model_checkpoint_path
            if epoch is not None:
                model_path = model_path.split('/')[:-1]
                model_path = '/'.join(model_path) + '/model.ckpt-' + str(epoch)
            saver.restore(sess, model_path)
            print("Model restored: " + model_path)
        else:
            raise ValueError('There are not any checkpoints.')

        print('Test Data Evaluation:')
        if params['label_type'] in ['character', 'character_capital_divide']:
            cer_clean_test = do_eval_cer(
                session=sess,
                decode_ops=[decode_op],
                network=network,
                dataset=test_clean_data,
                label_type=params['label_type'],
                eval_batch_size=params['batch_size'],
                progressbar=True)
            print('  CER (clean): %f %%' % (cer_clean_test * 100))

            cer_other_test = do_eval_cer(
                session=sess,
                decode_ops=[decode_op],
                network=network,
                dataset=test_other_data,
                label_type=params['label_type'],
                eval_batch_size=params['batch_size'],
                progressbar=True)
            print('  CER (other): %f %%' % (cer_other_test * 100))
        else:
            wer_clean_test = do_eval_wer(
                session=sess,
                decode_ops=[decode_op],
                network=network,
                dataset=test_clean_data,
                train_data_size=params['train_data_size'],
                is_test=True,
                eval_batch_size=params['batch_size'],
                progressbar=True)
            print('  WER (clean): %f %%' % (wer_clean_test * 100))

            wer_other_test = do_eval_wer(
                session=sess,
                decode_ops=[decode_op],
                network=network,
                dataset=test_other_data,
                train_data_size=params['train_data_size'],
                is_test=True,
                eval_batch_size=params['batch_size'],
                progressbar=True)
            print('  WER (other): %f %%' % (wer_other_test * 100))


def main(model_path, epoch):

    # Load config file
    with open(os.path.join(model_path, 'config.yml'), "r") as f:
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
            params['num_classes'] = 0

    # Model setting
    model = load(model_type=params['model'])
    network = model(
        input_size=params['input_size'] * params['num_stack'],
        num_unit=params['num_unit'],
        num_layer=params['num_layer'],
        num_classes=params['num_classes'],
        parameter_init=params['weight_init'],
        clip_grad=params['clip_grad'],
        clip_activation=params['clip_activation'],
        dropout_ratio_input=params['dropout_input'],
        dropout_ratio_hidden=params['dropout_hidden'],
        dropout_ratio_output=params['dropout_output'],
        num_proj=params['num_proj'],
        weight_decay=params['weight_decay'])

    network.model_dir = model_path
    do_eval(network=network, params=params, epoch=epoch)


if __name__ == '__main__':

    args = sys.argv
    if len(args) == 2:
        model_path = args[1]
        epoch = None
    elif len(args) == 3:
        model_path = args[1]
        epoch = args[2]
    else:
        raise ValueError(
            ("Set a path to saved model.\n"
             "Usase: python eval_ctc.py path_to_saved_model (epoch)"))
    main(model_path=model_path, epoch=epoch)
