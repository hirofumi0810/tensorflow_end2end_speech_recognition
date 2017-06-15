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
        dataset: Dataset class
        label_type: phone39 or phone48 or phone61 or character
        save_path: path to save decoding results
    """
    batch_size = 1
    num_examples = dataset.data_num
    iteration = int(num_examples / batch_size)
    if (num_examples / batch_size) != int(num_examples / batch_size):
        iteration += 1

    map_file_path_phone = '../metric/mapping_files/ctc/phone2num_' + \
        label_type[5:7] + '.txt'
    map_file_path_char = '../metric/mapping_files/ctc/char2num.txt'

    if save_path is not None:
        sys.stdout = open(join(network.model_dir, 'decode.txt'), 'w')

    for step in range(iteration):
        # Create feed dictionary for next mini batch
        inputs, labels_true, seq_len, input_names = dataset.next_batch(
            batch_size=batch_size)

        feed_dict = {
            network.inputs: inputs,
            network.seq_len: seq_len,
            network.keep_prob_input: 1.0,
            network.keep_prob_hidden: 1.0
        }

        # Visualize
        batch_size_each = len(labels_true)
        labels_pred_st = session.run(decode_op, feed_dict=feed_dict)
        labels_pred = sparsetensor2list(labels_pred_st, batch_size_each)
        for i_batch in range(batch_size_each):
            if label_type == 'character':
                print('----- wav: %s -----' % input_names[i_batch])
                print('True: %s' % num2char(
                    labels_true[i_batch], map_file_path_char))
                print('Pred: %s' % num2char(
                    labels_pred[i_batch], map_file_path_char))

            else:
                # Decode (mapped to 39 phones)
                print('----- wav: %s -----' % input_names[i_batch])
                print('True: %s' % num2phone(
                    labels_true[i_batch], map_file_path_phone))

                print('Pred: %s' % num2phone(
                    labels_pred[i_batch], map_file_path_phone))


def decode_test_multitask(session, decode_op_main, decode_op_second, network,
                          dataset, label_type_second, save_path=None):
    """Visualize label outputs of Multi-task CTC model.
    Args:
        session: session of training model
        decode_op_main: operation for decoding in the main task
        decode_op_second: operation for decoding in the second task
        network: network to evaluate
        dataset: Dataset class
        label_type_second: phone39 or phone48 or phone61
        save_path: path to save decoding results
    """
    batch_size = 1
    num_examples = dataset.data_num
    iteration = int(num_examples / batch_size)
    if (num_examples / batch_size) != int(num_examples / batch_size):
        iteration += 1

    if save_path is not None:
        sys.stdout = open(join(network.model_dir, 'decode.txt'), 'w')

    # Decode character
    map_file_path_char = '../metric/mapping_files/ctc/char2num.txt'
    for step in range(iteration):
        # Create feed dictionary for next mini batch
        inputs, labels_true, _, seq_len, input_names = dataset.next_batch(
            batch_size=batch_size)

        feed_dict = {
            network.inputs: inputs,
            network.seq_len: seq_len,
            network.keep_prob_input: 1.0,
            network.keep_prob_hidden: 1.0
        }

        # Visualize
        batch_size_each = len(labels_true)
        labels_pred_st = session.run(decode_op_main, feed_dict=feed_dict)
        labels_pred = sparsetensor2list(labels_pred_st, batch_size_each)
        for i_batch in range(batch_size_each):
            print('----- wav: %s -----' % input_names[i_batch])
            print('True: %s' % num2char(
                labels_true[i_batch], map_file_path_char))
            print('Pred: %s' % num2char(
                labels_pred[i_batch], map_file_path_char))

    # Decode phone
    map_file_path_phone = '../metric/mapping_files/ctc/phone2num_' + \
        label_type_second[5:7] + '.txt'
    for step in range(iteration):
        # Create feed dictionary for next mini batch
        inputs, _, labels_true, seq_len, input_names = dataset.next_batch(
            batch_size=batch_size)

        feed_dict = {
            network.inputs: inputs,
            network.seq_len: seq_len,
            network.keep_prob_input: 1.0,
            network.keep_prob_hidden: 1.0
        }

        # Visualize
        batch_size_each = len(labels_true)
        labels_pred_st = session.run(decode_op_second, feed_dict=feed_dict)
        labels_pred = sparsetensor2list(labels_pred_st, batch_size_each)
        for i_batch in range(batch_size_each):
            # Decode (mapped to 39 phones)
            print('----- wav: %s -----' % input_names[i_batch])
            print('True: %s' % num2phone(
                labels_true[i_batch], map_file_path_phone))

            print('Pred: %s' % num2phone(
                labels_pred[i_batch], map_file_path_phone))
