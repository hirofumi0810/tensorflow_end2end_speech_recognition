#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Define evaluation method for CTC network (CSJ corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import Levenshtein

from utils.labels.character import num2char
from utils.sparsetensor import sparsetensor2list
from utils.exception_func import exception
from utils.progressbar import wrap_iterator


@exception
def do_eval_per(session, per_op, network, dataset,
                eval_batch_size=None, is_progressbar=False,
                is_multitask=False):
    """Evaluate trained model by Phone Error Rate.
    Args:
        session: session of training model
        per_op: operation for computing phone error rate
        network: network to evaluate
        dataset: An instance of a `Dataset' class
        eval_batch_size: int, the batch size when evaluating the model
        is_progressbar: if True, visualize progressbar
        is_multitask: if True, evaluate the multitask model
    Returns:
        per_global: An average of PER
    """
    if eval_batch_size is None:
        batch_size = network.batch_size
    else:
        batch_size = eval_batch_size

    num_examples = dataset.data_num
    iteration = int(num_examples / batch_size)
    if (num_examples / batch_size) != int(num_examples / batch_size):
        iteration += 1
    per_global = 0

    # Make data generator
    mini_batch = dataset.next_batch(batch_size=batch_size)

    for step in wrap_iterator(range(iteration), is_progressbar):
        # Create feed dictionary for next mini batch
        if not is_multitask:
            inputs, labels_true_st, inputs_seq_len, _ = mini_batch.__next__()
        else:
            inputs, _, labels_true_st,  inputs_seq_len, _ = mini_batch.__next__()

        feed_dict = {
            network.inputs: inputs,
            network.inputs_seq_len: inputs_seq_len,
            network.keep_prob_input: 1.0,
            network.keep_prob_hidden: 1.0
        }

        batch_size_each = len(inputs_seq_len)

        per_local = session.run(per_op, feed_dict=feed_dict)
        per_global += per_local * batch_size_each

    per_global /= dataset.data_num

    return per_global


# @exception
def do_eval_cer(session, decode_op, network, dataset, label_type, is_test=None,
                eval_batch_size=None, is_progressbar=False,
                is_multitask=False, is_main=False):
    """Evaluate trained model by Character Error Rate.
    Args:
        session: session of training model
        decode_op: operation for decoding
        network: network to evaluate
        dataset: An instance of `Dataset` class
        label_type: string, character or kanji
        is_test: set to True when evaluating by the test set
        eval_batch_size: int, the batch size when evaluating the model
        is_progressbar: if True, visualize progressbar
        is_multitask: if True, evaluate the multitask model
        is_main: if True, evaluate the main task
    Return:
        cer_mean: An average of CER
    """
    if eval_batch_size is None:
        batch_size = network.batch_size
    else:
        batch_size = eval_batch_size

    num_examples = dataset.data_num
    iteration = int(num_examples / batch_size)
    if (num_examples / batch_size) != int(num_examples / batch_size):
        iteration += 1
    cer_sum = 0

    # Make data generator
    mini_batch = dataset.next_batch(batch_size=batch_size)

    if label_type == 'character':
        map_file_path = '../metric/mapping_files/ctc/char2num.txt'
    elif label_type == 'kanji':
        map_file_path = '../metric/mapping_files/ctc/kanji2num.txt'
    for step in wrap_iterator(range(iteration), is_progressbar):
        # Create feed dictionary for next mini batch
        if not is_multitask:
            inputs, labels_true_st, inputs_seq_len, _ = mini_batch.__next__()
        else:
            if is_main:
                inputs, labels_true_st, _, inputs_seq_len, _ = mini_batch.__next__()
            else:
                inputs, _, labels_true_st, inputs_seq_len, _ = mini_batch.__next__()

        feed_dict = {
            network.inputs: inputs,
            network.inputs_seq_len: inputs_seq_len,
            network.keep_prob_input: 1.0,
            network.keep_prob_hidden: 1.0
        }

        batch_size_each = len(inputs_seq_len)

        labels_pred_st = session.run(decode_op, feed_dict=feed_dict)
        labels_true = sparsetensor2list(labels_true_st, batch_size_each)
        labels_pred = sparsetensor2list(labels_pred_st, batch_size_each)
        for i_batch in range(batch_size_each):
            # Convert from list to string
            str_pred = num2char(labels_pred[i_batch], map_file_path)
            # TODO: change in case of character
            if label_type == 'kanji' and is_test:
                str_true = ''.join(labels_true[i_batch])
                # NOTE* 漢字の場合はテストデータのラベルはそのまま保存してある
            else:
                str_true = num2char(labels_true[i_batch], map_file_path)

            # Remove silence(_) labels
            str_true = re.sub(r'[_]+', "", str_true)
            str_pred = re.sub(r'[_]+', "", str_pred)

            # Compute edit distance
            cer_each = Levenshtein.distance(
                str_pred, str_true) / len(list(str_true))

            cer_sum += cer_each

    cer_mean = cer_sum / dataset.data_num

    return cer_mean
