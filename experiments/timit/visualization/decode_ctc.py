#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Decode the trained CTC outputs (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf
import yaml

sys.path.append('../../../')
from experiments.timit.data.load_dataset_ctc import Dataset
from experiments.timit.visualization.core.decode.ctc import decode_test
from models.ctc.load_model import load


def do_decode(network, params, epoch=None):
    """Decode the CTC outputs.
    Args:
        network: model to restore
        params: A dictionary of parameters
        epoch: int, the epoch to restore
    """
    # Load dataset
    test_data = Dataset(
        data_type='test', label_type=params['label_type'],
        batch_size=1, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=False, progressbar=True)

    # Define placeholders
    network.create_placeholders()

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

        # Visualize
        decode_test(session=sess,
                    decode_op=decode_op,
                    network=network,
                    dataset=test_data,
                    label_type=params['label_type'],
                    save_path=None)
        # save_path=network.model_dir)


def main(model_path, epoch):

    # Load config file
    with open(os.path.join(model_path, 'config.yml'), "r") as f:
        config = yaml.load(f)
        params = config['param']

    # Except for a blank label
    if params['label_type'] == 'phone61':
        params['num_classes'] = 61
    elif params['label_type'] == 'phone48':
        params['num_classes'] = 48
    elif params['label_type'] == 'phone39':
        params['num_classes'] = 39
    elif params['label_type'] == 'character':
        params['num_classes'] = 28
    elif params['label_type'] == 'character_capital_divide':
        params['num_classes'] = 72

    # Model setting
    model = load(model_type=params['model'])
    network = model(
        input_size=params['input_size'] * params['num_stack'],
        num_units=params['num_units'],
        num_layers=params['num_layers'],
        num_classes=params['num_classes'],
        bidirectional=params['bidirectional'],
        parameter_init=params['weight_init'],
        clip_grad=params['clip_grad'],
        clip_activation=params['clip_activation'],
        num_proj=params['num_proj'],
        weight_decay=params['weight_decay'])

    network.model_dir = model_path
    do_decode(network=network, params=params, epoch=epoch)


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
             "Usase: python decode_ctc.py path_to_saved_model"))
    main(model_path=model_path, epoch=epoch)
