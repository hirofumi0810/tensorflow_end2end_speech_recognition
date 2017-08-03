#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Define evaluation method for CTC network (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import Levenshtein

from experiments.utils.data.labels.character import num2char
from experiments.utils.data.labels.word import num2word
from experiments.utils.data.sparsetensor import list2sparsetensor, sparsetensor2list
from experiments.utils.evaluation.edit_distance import compute_edit_distance
from experiments.utils.progressbar import wrap_generator


def do_eval_cer(session, decode_ops, network, dataset, label_type,
                eval_batch_size=None, progressbar=False, is_multitask=False):
    """Evaluate trained model by Character Error Rate.
    Args:
        session: session of training model
        decode_ops: list of operations for decoding
        network: network to evaluate
        dataset: An instance of a `Dataset` class
        label_type: string, character or character_capital_divide
        eval_batch_size: int, the batch size when evaluating the model
        progressbar: if True, visualize the progressbar
        is_multitask: if True, evaluate the multitask model
    Return:
        cer_mean: An average of CER
    """
    assert isinstance(decode_ops, list), "decode_ops must be a list."

    # Reset data counter
    dataset.reset()

    batch_size = dataset.batch_size if eval_batch_size is None else eval_batch_size

    if label_type == 'character':
        map_file_path = '../metrics/mapping_files/ctc/character2num.txt'
    else:
        map_file_path = '../metrics/mapping_files/ctc/character2num_capital.txt'

    cer_mean = 0
    skip_data_num = 0
    total_step = int(dataset.data_num / batch_size)
    if (dataset.data_num / batch_size) != int(dataset.data_num / batch_size):
        total_step += 1
    for data, next_epoch_flag in wrap_generator(dataset(batch_size),
                                                progressbar,
                                                total=total_step):
        # Create feed dictionary for next mini batch
        if is_multitask:
            inputs, _, labels_true, inputs_seq_len, _ = data
        else:
            inputs, labels_true, inputs_seq_len, _ = data

        feed_dict = {}
        for i_device in range(len(decode_ops)):
            feed_dict[network.inputs_pl_list[i_device]] = inputs[i_device]
            feed_dict[network.inputs_seq_len_pl_list[i_device]
                      ] = inputs_seq_len[i_device]
            feed_dict[network.keep_prob_input_pl_list[i_device]
                      ] = 1.0
            feed_dict[network.keep_prob_hidden_pl_list[i_device]
                      ] = 1.0
            feed_dict[network.keep_prob_output_pl_list[i_device]
                      ] = 1.0

        labels_pred_st_list = session.run(decode_ops, feed_dict=feed_dict)
        for i_device, labels_pred_st in enumerate(labels_pred_st_list):
            batch_size_device = len(inputs_seq_len[i_device])
            try:
                labels_pred = sparsetensor2list(labels_pred_st,
                                                batch_size_device)
                for i_batch in range(batch_size_device):
                    # Convert from list to string
                    str_true = num2char(
                        labels_true[i_device][i_batch], map_file_path)
                    str_pred = num2char(labels_pred[i_batch], map_file_path)

                    # Remove silence(_) labels
                    str_true = re.sub(r'[_\']+', "", str_true)
                    str_pred = re.sub(r'[_\']+', "", str_pred)

                    # Convert to lower case
                    if label_type == 'character_capital_divide':
                        str_true = str_true.lower()
                        str_pred = str_pred.lower()

                    # Compute edit distance
                    cer_mean = Levenshtein.distance(
                        str_pred, str_true) / len(list(str_true))

            except IndexError:
                skip_data_num += batch_size_device
                # TODO: Conduct decoding again with batch size 1

        if next_epoch_flag:
            break

    cer_mean /= (dataset.data_num - batch_size_device)

    return cer_mean


def do_eval_wer(session, decode_ops, network, dataset, train_data_size, is_test,
                eval_batch_size=None, progressbar=False, is_multitask=False):
    """Evaluate trained model by Word Error Rate.
    Args:
        session: session of training model
        decode_ops: list of operations for decoding
        network: network to evaluate
        dataset: An instance of `Dataset` class
        train_data_size: string, train_clean100 or train_clean360 or
            train_other500 or train_all
        is_test: bool, set to True when evaluating by the test set
        eval_batch_size: int, the batch size when evaluating the model
        progressbar: if True, visualize progressbar
        is_multitask: if True, evaluate the multitask model
    Return:
        wer_mean: An average of WER
    """
    assert isinstance(decode_ops, list), "decode_ops must be a list."

    # Reset data counter
    dataset.reset()

    batch_size = dataset.batch_size if eval_batch_size is None else eval_batch_size

    # map_file_path = '../metrics/mapping_files/ctc/word2num_' +
    # train_data_size + '.txt'

    wer_mean = 0
    skip_data_num = 0
    total_step = int(dataset.data_num / batch_size)
    if (dataset.data_num / batch_size) != int(dataset.data_num / batch_size):
        total_step += 1
    for data, next_epoch_flag in wrap_generator(dataset(batch_size),
                                                progressbar,
                                                total=total_step):
        # Create feed dictionary for next mini batch
        if is_multitask:
            inputs, labels_true, _, inputs_seq_len, _ = data
        else:
            inputs, labels_true, inputs_seq_len, _ = data

        feed_dict = {}
        for i_device in range(len(decode_ops)):
            feed_dict[network.inputs_pl_list[i_device]] = inputs[i_device]
            feed_dict[network.inputs_seq_len_pl_list[i_device]
                      ] = inputs_seq_len[i_device]
            feed_dict[network.keep_prob_input_pl_list[i_device]
                      ] = 1.0
            feed_dict[network.keep_prob_hidden_pl_list[i_device]
                      ] = 1.0
            feed_dict[network.keep_prob_output_pl_list[i_device]
                      ] = 1.0

        labels_pred_st_list = session.run(decode_ops, feed_dict=feed_dict)
        for i_device, labels_pred_st in enumerate(labels_pred_st_list):
            batch_size_device = len(inputs_seq_len[i_device])
            try:
                labels_pred = sparsetensor2list(labels_pred_st,
                                                batch_size_device)

                # Compute edit distance
                labels_true_st = list2sparsetensor(labels_true,
                                                   padded_value=-1)
                labels_pred_st = list2sparsetensor(labels_pred,
                                                   padded_value=-1)
                wer_mean += compute_edit_distance(session,
                                                  labels_true_st,
                                                  labels_pred_st)
            except IndexError:
                skip_data_num += batch_size_device
                # TODO: Conduct decoding again with batch size 1

        if next_epoch_flag:
            break

    wer_mean /= (dataset.data_num - skip_data_num)
    # TODO: This is just edit distance.

    return wer_mean
