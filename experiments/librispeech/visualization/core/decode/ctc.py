#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilities for decoding of the CTC model (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import sys

from experiments.utils.data.labels.character import num2char
from experiments.utils.data.labels.word import num2word
from experiments.utils.data.sparsetensor import sparsetensor2list


def decode_test(session, decode_op, network, dataset, label_type,
                save_path=None):
    """Visualize label outputs of CTC model.
    Args:
        session: session of training model
        decode_op: operation for decoding
        network: network to evaluate
        dataset: An instance of a `Dataset` class
        label_type: string,  character or character_capital_divide or word
        save_path: path to save decoding results
    """
    if label_type == 'character':
        map_file_path = '../metrics/mapping_files/ctc/character2num.txt'
    elif label_type == 'character_capital_divide':
        map_file_path = '../metrics/mapping_files/ctc/character2num_capital.txt'
    elif label_type == 'word':
        raise NotImplementedError

    if save_path is not None:
        sys.stdout = open(join(network.model_dir, 'decode.txt'), 'w')

    # Batch size is expected to be 1
    for data, next_epoch_flag in dataset(batch_size=1):
        # Create feed dictionary for next mini batch
        inputs, labels_true, inputs_seq_len, input_names = data

        feed_dict = {
            network.inputs_pl_list[0]: inputs[0],
            network.inputs_seq_len_pl_list[0]: inputs_seq_len[0],
            network.keep_prob_input_pl_list[0]: 1.0,
            network.keep_prob_hidden_pl_list[0]: 1.0,
            network.keep_prob_output_pl_list[0]: 1.0
        }

        # Visualize
        labels_pred_st = session.run(decode_op, feed_dict=feed_dict)
        labels_pred = sparsetensor2list(labels_pred_st, batch_size=1)

        if label_type in ['character', 'character_capital_divide']:
            print('----- wav: %s -----' % input_names[0])
            print('True: %s' % num2char(
                labels_true[0], map_file_path))
            print('Pred: %s' % num2char(
                labels_pred[0], map_file_path))

        else:
            print('----- wav: %s -----' % input_names[0])
            print('True: %s' % num2word(
                labels_true[0], map_file_path))
            print('Pred: %s' % num2word(
                labels_pred[0], map_file_path))

        if next_epoch_flag:
            break
