#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate the trained multi-task CTC model (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf
import yaml

sys.path.append('../../../')
from experiments.timit.data.load_dataset_multitask_ctc import Dataset
from experiments.timit.metrics.ctc import do_eval_per, do_eval_cer
from models.ctc.load_model_multitask import load


def do_eval(network, params, epoch=None):
    """Evaluate the model.
    Args:
        network: model to restore
        params: A dictionary of parameters
        epoch: int, the epoch to restore
    """
    # Load dataset
    test_data = Dataset(
        data_type='test', label_type_main=params['label_type_main'],
        label_type_sub='phone39', batch_size=1,
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=False, progressbar=True)

    # Define placeholders
    network.create_placeholders(gpu_index=None)

    # Add to the graph each operation
    _, logits_main, logits_sub = network.compute_loss(
        network.inputs_pl_list[0],
        network.labels_pl_list[0],
        network.labels_sub_pl_list[0],
        network.inputs_seq_len_pl_list[0],
        network.keep_prob_input_pl_list[0],
        network.keep_prob_hidden_pl_list[0],
        network.keep_prob_output_pl_list[0])
    decode_op_main, decode_op_sub = network.decoder(
        logits_main,
        logits_sub,
        network.inputs_seq_len_pl_list[0],
        decode_type='beam_search',
        beam_width=20)
    _, per_op = network.compute_ler(
        decode_op_main, decode_op_sub,
        network.labels_pl_list[0], network.labels_sub_pl_list[0])

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

        print('=== Test Data Evaluation ===')
        cer_test = do_eval_cer(
            session=sess,
            decode_op=decode_op_main,
            network=network,
            dataset=test_data,
            label_type=params['label_type_main'],
            eval_batch_size=1,
            progressbar=True,
            is_multitask=True)
        print('  CER: %f %%' % (cer_test * 100))

        per_test = do_eval_per(
            session=sess,
            decode_op=decode_op_sub,
            per_op=per_op,
            network=network,
            dataset=test_data,
            label_type=params['label_type_sub'],
            eval_batch_size=1,
            progressbar=True,
            is_multitask=True)
        print('  PER: %f %%' % (per_test * 100))


def main(model_path, epoch):

    # Load config file
    with open(os.path.join(model_path, 'config.yml'), "r") as f:
        config = yaml.load(f)
        params = config['param']

    # Except for a blank label
    if params['label_type_main'] == 'character':
        params['num_classes_main'] = 28
    elif params['label_type_main'] == 'character_capital_divide':
        params['num_classes_main'] = 72

    if params['label_type_sub'] == 'phone61':
        params['num_classes_sub'] = 61
    elif params['label_type_sub'] == 'phone48':
        params['num_classes_sub'] = 48
    elif params['label_type_sub'] == 'phone39':
        params['num_classes_sub'] = 39

    # Model setting
    model = load(model_type=params['model'])
    network = model(
        input_size=params['input_size'] * params['num_stack'],
        num_unit=params['num_unit'],
        num_layer_main=params['num_layer_main'],
        num_layer_sub=params['num_layer_sub'],
        num_classes_main=params['num_classes_main'],
        num_classes_sub=params['num_classes_sub'],
        main_task_weight=params['main_task_weight'],
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
             "Usase: python eval_multitask_ctc.py path_to_saved_model (epoch)"))

    main(model_path=args[1], epoch=epoch)
