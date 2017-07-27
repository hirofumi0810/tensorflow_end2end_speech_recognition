#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilities for decoding of the CTC model (CSJ corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import sys

from experiments.utils.labels.character import num2char
from experiments.utils.labels.phone import num2phone
from experiments.utils.sparsetensor import sparsetensor2list


def decode_test(session, decode_op, network, dataset, label_type,
                save_path=None):
    """Visualize label outputs of CTC model.
    Args:
        session: session of training model
        decode_op: operation for decoding
        network: network to evaluate
        dataset: An instance of a `Dataset` class
        label_type: string, kanji or kana or phone
        save_path: path to save decoding results
    """
    # Batch size is expected to be 1
    iteration = dataset.data_num

    # Make data generator
    mini_batch = dataset.next_batch(batch_size=1)

    if label_type == 'kanji':
        map_file_path = '../../../metrics/mapping_files/ctc/kanji2num.txt'
    elif label_type == 'kana':
        map_file_path = '../../../metrics/mapping_files/ctc/kana2num.txt'
    elif label_type == 'phone':
        map_file_path = '../../../metrics/mapping_files/ctc/phone2num.txt'

    # if save_path is not None:
    #     sys.stdout = open(join(network.model_dir, 'decode.txt'), 'w')

    for step in range(iteration):
        # Create feed dictionary for next mini batch
        inputs, labels_true, inputs_seq_len, input_names = mini_batch.__next__()
        # NOTE: labels_true is expected to be a list of string when evaluation
        # using dataset where label_type is kanji or kana

        if input_names[0] not in ['A03M0106_0057', 'A03M0016_0014']:
            continue

        feed_dict = {
            network.inputs: inputs,
            network.inputs_seq_len: inputs_seq_len,
            network.keep_prob_input: 1.0,
            network.keep_prob_hidden: 1.0,
            network.keep_prob_output: 1.0
        }

        # Visualize
        labels_pred_st = session.run(decode_op, feed_dict=feed_dict)
        labels_pred = sparsetensor2list(labels_pred_st, batch_size=1)

        if label_type in ['kanji', 'kana']:
            print('----- wav: %s -----' % input_names[0])
            print('True: %s' % labels_true[0])
            print('Pred: %s' % num2char(labels_pred[0], map_file_path))

        elif label_type == 'phone':
            print('----- wav: %s -----' % input_names[0])
            print('True: %s' % num2phone(labels_true[0], map_file_path))
            print('Pred: %s' % num2phone(labels_pred[0], map_file_path))
