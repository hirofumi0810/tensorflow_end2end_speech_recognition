#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Define evaluation method for CTC network (CSJ corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import numpy as np
import Levenshtein
from tqdm import tqdm

from utils.labels.character import num2char
from utils.labels.phone import num2phone
from utils.data.sparsetensor import list2sparsetensor, sparsetensor2list
from utils.exception_func import exception


@exception
def do_eval_per(session, per_op, network, dataset,
                eval_batch_size=None, rate=1.0, is_progressbar=False,
                is_multitask=False):
    """Evaluate trained model by Phone Error Rate.
    Args:
        session: session of training model
        per_op: operation for computing phone error rate
        network: network to evaluate
        dataset: `Dataset' class
        eval_batch_size: batch size on evaluation
        rate: A float value. Rate of evaluation data to use
        is_progressbar: if True, evaluate during training, else during restoring
        is_multitask: if True, evaluate the multitask model
    Returns:
        per_global: phone error rate
    """
    batch_size = network.batch_size if eval_batch_size is None else eval_batch_size

    num_examples = dataset.data_num * rate
    iteration = int(num_examples / batch_size)
    if (num_examples / batch_size) != int(num_examples / batch_size):
        iteration += 1
    per_global = 0

    iterator = tqdm(range(iteration)) if is_progressbar else range(iteration)
    for step in iterator:
        # Create feed dictionary for next mini batch
        if not is_multitask:
            inputs, labels_true, seq_len, _ = dataset.next_batch(
                batch_size=batch_size)
        else:
            inputs, _, labels_true,  seq_len, _ = dataset.next_batch(
                batch_size=batch_size)

        feed_dict = {
            network.inputs_pl: inputs,
            network.seq_len_pl: seq_len,
            network.keep_prob_input_pl: 1.0,
            network.keep_prob_hidden_pl: 1.0
        }

        batch_size_each = len(labels_true)

        per_local = session.run(per_op, feed_dict=feed_dict)
        per_global += per_local * batch_size_each

    per_global /= dataset.data_num
    print('  PER: %f' % per_global)

    return per_global


@exception
def do_eval_cer(session, decode_op, network, dataset, label_type, is_test=None,
                eval_batch_size=None, rate=1.0, is_progressbar=False,
                is_multitask=False, is_main=False):
    """Evaluate trained model by Character Error Rate.
    Args:
        session: session of training model
        decode_op: operation for decoding
        network: network to evaluate
        dataset: Dataset class
        label_type: character or kanji
        is_test: set to True when evaluating by the test set
        eval_batch_size: batch size on evaluation
        rate: rate of evaluation data to use
        is_progressbar: if True, visualize progressbar
        is_multitask: if True, evaluate the multitask model
        is_main: if True, evaluate the main task
    Return:
        cer_mean: mean character error rate
    """
    batch_size = network.batch_size if eval_batch_size is None else eval_batch_size

    num_examples = dataset.data_num * rate
    iteration = int(num_examples / batch_size)
    if (num_examples / batch_size) != int(num_examples / batch_size):
        iteration += 1
    cer_sum = 0

    if label_type == 'character':
        map_file_path = '../evaluation/mapping_files/ctc/char2num.txt'
    elif label_type == 'kanji':
        map_file_path = '../evaluation/mapping_files/ctc/kanji2num.txt'
    iterator = tqdm(range(iteration)) if is_progressbar else range(iteration)
    for step in iterator:
        # Create feed dictionary for next mini batch
        if not is_multitask:
            inputs, labels_true, seq_len, _ = dataset.next_batch(
                batch_size=batch_size)
        else:
            if is_main:
                inputs, labels_true, _, seq_len, _ = dataset.next_batch(
                    batch_size=batch_size)
            else:
                inputs, _, labels_true, seq_len, _ = dataset.next_batch(
                    batch_size=batch_size)

        feed_dict = {
            network.inputs_pl: inputs,
            network.seq_len_pl: seq_len,
            network.keep_prob_input_pl: 1.0,
            network.keep_prob_hidden_pl: 1.0
        }

        batch_size_each = len(labels_true)
        labels_pred_st = session.run(decode_op, feed_dict=feed_dict)
        labels_pred = sparsetensor2list(labels_pred_st, batch_size_each)
        for i_batch in range(batch_size_each):

            # Convert from list to string
            str_pred = num2char(labels_pred[i_batch], map_file_path)
            str_pred = re.sub(r'_', '', str_pred)
            # TODO: change in case of character
            if label_type == 'kanji' and is_test:
                str_true = labels_true[i_batch]
            else:
                str_true = num2char(labels_true[i_batch], map_file_path)
            str_true = re.sub(r'_', '', str_true)

            # Compute edit distance
            cer_each = Levenshtein.distance(
                str_pred, str_true) / len(list(str_true))
            cer_sum += cer_each

    cer_mean = cer_sum / dataset.data_num
    print('  CER: %f %%' % (cer_mean * 100))

    return cer_mean


def decode_test(session, decode_op, network, dataset, label_type, is_test,
                eval_batch_size=None, rate=1.0):
    """Visualize label outputs.
    Args:
        session: session of training model
        decode_op: operation for decoding
        network: network to evaluate
        dataset: Dataset class
        label_type: phone or character or kanji
        is_test: set to True when evaluating by the test set
        eval_batch_size: batch size on evaluation
        rate: rate of evaluation data to use
    """
    batch_size = 1
    num_examples = dataset.data_num * rate
    iteration = int(num_examples / batch_size)
    if (num_examples / batch_size) != int(num_examples / batch_size):
        iteration += 1

    map_file_path_phone = '../evaluation/mapping_files/ctc/phone2num.txt'
    if label_type == 'character':
        map_file_path = '../evaluation/mapping_files/ctc/char2num.txt'
    elif label_type == 'kanji':
        map_file_path = '../evaluation/mapping_files/ctc/kanji2num.txt'
    for step in range(iteration):
        # Create feed dictionary for next mini batch
        inputs, labels_true, seq_len, input_names = dataset.next_batch(
            batch_size=batch_size)

        feed_dict = {
            network.inputs_pl: inputs,
            network.seq_len_pl: seq_len,
            network.keep_prob_input_pl: 1.0,
            network.keep_prob_hidden_pl: 1.0
        }

        # Visualize
        batch_size_each = len(labels_true)
        labels_pred_st = session.run(decode_op, feed_dict=feed_dict)
        labels_pred = sparsetensor2list(labels_pred_st, batch_size_each)
        for i_batch in range(batch_size_each):
            if label_type in ['character', 'kanji']:
                # Convert from list to string
                str_pred = num2char(labels_pred[i_batch], map_file_path)
                if label_type == 'kanji' and is_test:
                    str_true = labels_true[i_batch]
                else:
                    str_true = num2char(labels_true[i_batch], map_file_path)

                print('-----wav: %s-----' % input_names[i_batch])
                print('True: %s' % str_true)
                print('Pred: %s' % str_pred)

            elif label_type == 'phone':
                print('-----wav: %s-----' % input_names[i_batch])
                print('True: %s' % num2phone(
                    labels_true[i_batch], map_file_path_phone))
                print('Pred: %s' % num2phone(
                    labels_pred[i_batch], map_file_path_phone))
