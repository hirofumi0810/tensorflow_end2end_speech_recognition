#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilities for decoding of CTC model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import sys

from utils.labels.character import num2char
from utils.labels.phone import num2phone
from utils.sparsetensor import sparsetensor2list


def decode_test(session, decode_op, network, dataset, label_type,
                save_path=None):
    """Visualize label outputs of CTC model.
    Args:
        session: session of training model
        decode_op: operation for decoding
        network: network to evaluate
        dataset: An instance of a `Dataset` class
        label_type: string, phone39 or phone48 or phone61 or character
        save_path: path to save decoding results
    """
    # Batch size is expected to be 1
    iteration = dataset.data_num

    if label_type == 'character':
        map_file_path = '../metric/mapping_files/ctc/char2num.txt'
    else:
        map_file_path = '../metric/mapping_files/ctc/phone2num_' + \
            label_type[5:7] + '.txt'

    if save_path is not None:
        sys.stdout = open(join(network.model_dir, 'decode.txt'), 'w')

    for step in range(iteration):
        # Create feed dictionary for next mini batch
        inputs, labels_true_st, inputs_seq_len, input_names = dataset.next_batch()

        feed_dict = {
            network.inputs: inputs,
            network.inputs_seq_len: inputs_seq_len,
            network.keep_prob_input: 1.0,
            network.keep_prob_hidden: 1.0
        }

        # Visualize
        labels_pred_st = session.run(decode_op, feed_dict=feed_dict)
        labels_true = sparsetensor2list(labels_true_st, batch_size=1)
        labels_pred = sparsetensor2list(labels_pred_st, batch_size=1)

        if label_type == 'character':
            print('----- wav: %s -----' % input_names[0])
            print('True: %s' % num2char(
                labels_true[0], map_file_path))
            print('Pred: %s' % num2char(
                labels_pred[0], map_file_path))

        else:
            # Decode (mapped to 39 phones)
            print('----- wav: %s -----' % input_names[0])
            print('True: %s' % num2phone(
                labels_true[0], map_file_path))

            print('Pred: %s' % num2phone(
                labels_pred[0], map_file_path))


def decode_test_multitask(session, decode_op_main, decode_op_second, network,
                          dataset, label_type_second, save_path=None):
    """Visualize label outputs of Multi-task CTC model.
    Args:
        session: session of training model
        decode_op_main: operation for decoding in the main task
        decode_op_second: operation for decoding in the second task
        network: network to evaluate
        dataset: An instance of a `Dataset` class
        label_type_second: string, phone39 or phone48 or phone61
        save_path: path to save decoding results
    """
    # Batch size is expected to be 1
    iteration = dataset.data_num

    if save_path is not None:
        sys.stdout = open(join(network.model_dir, 'decode.txt'), 'w')

    # Decode character
    print('===== character =====')
    map_file_path = '../metric/mapping_files/ctc/char2num.txt'
    for step in range(iteration):
        # Create feed dictionary for next mini batch
        inputs, labels_true_st, _, inputs_seq_len, input_names = dataset.next_batch()

        feed_dict = {
            network.inputs: inputs,
            network.inputs_seq_len: inputs_seq_len,
            network.keep_prob_input: 1.0,
            network.keep_prob_hidden: 1.0
        }

        # Visualize
        labels_pred_st = session.run(decode_op_main, feed_dict=feed_dict)
        labels_true = sparsetensor2list(labels_true_st, batch_size=1)
        labels_pred = sparsetensor2list(labels_pred_st, batch_size=1)

        # Decode
        print('----- wav: %s -----' % input_names[0])
        print('True: %s' % num2char(
            labels_true[0], map_file_path))
        print('Pred: %s' % num2char(
            labels_pred[0], map_file_path))

    # Decode phone
    print('\n===== phone =====')
    map_file_path = '../metric/mapping_files/ctc/phone2num_' + \
        label_type_second[5:7] + '.txt'
    for step in range(iteration):
        # Create feed dictionary for next mini batch
        inputs, _, labels_true_st, inputs_seq_len, input_names = dataset.next_batch()

        feed_dict = {
            network.inputs: inputs,
            network.inputs_seq_len: inputs_seq_len,
            network.keep_prob_input: 1.0,
            network.keep_prob_hidden: 1.0
        }

        # Visualize
        labels_pred_st = session.run(decode_op_second, feed_dict=feed_dict)
        labels_true = sparsetensor2list(labels_true_st, batch_size=1)
        labels_pred = sparsetensor2list(labels_pred_st, batch_size=1)

        # Decode (mapped to 39 phones)
        print('----- wav: %s -----' % input_names[0])
        print('True: %s' % num2phone(
            labels_true[0], map_file_path))

        print('Pred: %s' % num2phone(
            labels_pred[0], map_file_path))
