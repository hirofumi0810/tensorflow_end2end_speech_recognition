#! /usr/bin/env python
# -*- coding: utf-8 -*-

""""Utilities for decoding of the Attention model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.labels.character import num2char
from utils.labels.phone import num2phone


def decode_test(session, decode_op, network, dataset, label_type,
                save_path=None):
    """Visualize label outputs.
    Args:
        session: session of training model
        decode_op: operation for decoding
        network: network to evaluate
        dataset: An instance of a `Dataset` class
        label_type: stirng, phone39 or phone48 or phone61 or character
        save_path: path to save decoding results
    """
    # Batch size is expected to be 1
    iteration = dataset.data_num

    # Make data generator
    mini_batch = dataset.next_batch(batch_size=1)

    if label_type == 'character':
        map_file_path = '../metrics/mapping_files/attention/char2num.txt'
    else:
        map_file_path = '../metrics/mapping_files/attention/phone2num_' + \
            label_type[5:7] + '.txt'

    # if save_path is not None:
    #     sys.stdout = open(join(network.model_dir, 'decode.txt'), 'w')

    for step in range(iteration):
        # Create feed dictionary for next mini batch
        inputs, labels_true, inputs_seq_len, _, input_names = mini_batch.__next__()

        feed_dict = {
            network.inputs: inputs,
            network.labels: labels_true,
            network.inputs_seq_len: inputs_seq_len,
            network.keep_prob_input: 1.0,
            network.keep_prob_hidden: 1.0
        }

        # Visualize
        labels_pred = session.run(decode_op, feed_dict=feed_dict)

        if label_type == 'character':
            print('----- wav: %s -----' % input_names[0])
            print('True: %s' % num2char(
                labels_true[0][1:-1], map_file_path))
            print('Pred: %s' % num2char(
                labels_pred[0], map_file_path).replace('>', ''))

        else:
            print('----- wav: %s -----' % input_names[0])
            print('True: %s' % num2phone(
                labels_true[0][1:-1], map_file_path))

            print('Pred: %s' % num2phone(
                labels_pred[0], map_file_path).replace('>', ''))
