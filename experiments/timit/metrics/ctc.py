#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Define evaluation method for CTC network (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import Levenshtein
import numpy as np

from experiments.timit.metrics.mapping import map_to_39phone
from experiments.utils.data.labels.character import num2char
from experiments.utils.data.labels.phone import num2phone, phone2num
from experiments.utils.data.sparsetensor import list2sparsetensor, sparsetensor2list
from experiments.utils.evaluation.edit_distance import compute_edit_distance
from experiments.utils.progressbar import wrap_generator


def do_eval_per(session, decode_op, per_op, network, dataset, label_type,
                eval_batch_size=None, progressbar=False,
                is_multitask=False):
    """Evaluate trained model by Phone Error Rate.
    Args:
        session: session of training model
        decode_op: operation for decoding
        per_op: operation for computing phone error rate
        network: network to evaluate
        dataset: An instance of a `Dataset' class
        label_type: string, phone39 or phone48 or phone61
        eval_batch_size: int, the batch size when evaluating the model
        progressbar: if True, visualize the progressbar
        is_multitask: if True, evaluate the multitask model
    Returns:
        per_mean: An average of PER
    """
    # Reset data counter
    dataset.reset()

    batch_size = dataset.batch_size if eval_batch_size is None else eval_batch_size
    train_label_type = label_type
    eval_label_type = dataset.label_type_sub if is_multitask else dataset.label_type

    train_phone2num_map_file_path = '../metrics/mapping_files/ctc/' + \
        train_label_type + '_to_num.txt'
    eval_phone2num_map_file_path = '../metrics/mapping_files/ctc/' + \
        eval_label_type + '_to_num.txt'
    phone2num_39_map_file_path = '../metrics/mapping_files/ctc/phone39_to_num.txt'
    phone2phone_map_file_path = '../metrics/mapping_files/phone2phone.txt'
    per_mean = 0
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

        feed_dict = {
            network.inputs_pl_list[0]: inputs,
            network.inputs_seq_len_pl_list[0]: inputs_seq_len,
            network.keep_prob_input_pl_list[0]: 1.0,
            network.keep_prob_hidden_pl_list[0]: 1.0,
            network.keep_prob_output_pl_list[0]: 1.0
        }

        batch_size_each = len(inputs_seq_len)

        # Evaluate by 39 phones
        labels_pred_st = session.run(decode_op, feed_dict=feed_dict)
        labels_pred = sparsetensor2list(labels_pred_st, batch_size_each)

        labels_pred_mapped, labels_true_mapped = [], []
        for i_batch in range(batch_size_each):
            ###############
            # Hypothesis
            ###############
            # Convert from num to phone (-> list of phone strings)
            phone_pred_list = num2phone(
                labels_pred[i_batch],
                train_phone2num_map_file_path).split(' ')

            # Mapping to 39 phones (-> list of phone strings)
            phone_pred_list = map_to_39phone(phone_pred_list,
                                             train_label_type,
                                             phone2phone_map_file_path)

            # Convert from phone to num (-> list of phone indices)
            phone_pred_list = phone2num(phone_pred_list,
                                        phone2num_39_map_file_path)
            labels_pred_mapped.append(phone_pred_list)

            ###############
            # Reference
            ###############
            # Convert from num to phone (-> list of phone strings)
            phone_true_list = num2phone(
                labels_true[i_batch],
                eval_phone2num_map_file_path).split(' ')

            # Mapping to 39 phones (-> list of phone strings)
            phone_true_list = map_to_39phone(phone_true_list,
                                             eval_label_type,
                                             phone2phone_map_file_path)

            # Convert from phone to num (-> list of phone indices)
            phone_true_list = phone2num(phone_true_list,
                                        phone2num_39_map_file_path)
            labels_true_mapped.append(phone_true_list)

        # Compute edit distance
        labels_true_st = list2sparsetensor(labels_true_mapped, padded_value=-1)
        labels_pred_st = list2sparsetensor(labels_pred_mapped, padded_value=-1)
        per_list = compute_edit_distance(session, labels_true_st,
                                         labels_pred_st)
        per_mean += np.sum(per_list)

        if next_epoch_flag:
            break

    per_mean /= dataset.data_num

    return per_mean


def do_eval_cer(session, decode_op, network, dataset, label_type,
                eval_batch_size=None, progressbar=False, is_multitask=False):
    """Evaluate trained model by Character Error Rate.
    Args:
        session: session of training model
        decode_op: operation for decoding
        network: network to evaluate
        dataset: An instance of a `Dataset` class
        label_type: string, character or character_capital_divide
        eval_batch_size: int, the batch size when evaluating the model
        progressbar: if True, visualize the progressbar
        is_multitask: if True, evaluate the multitask model
    Return:
        cer_mean: An average of CER
    """
    # Reset data counter
    dataset.reset()

    batch_size = dataset.batch_size if eval_batch_size is None else eval_batch_size

    map_file_path = '../metrics/mapping_files/ctc/' + label_type + '_to_num.txt'
    cer_mean = 0
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

        feed_dict = {
            network.inputs_pl_list[0]: inputs,
            network.inputs_seq_len_pl_list[0]: inputs_seq_len,
            network.keep_prob_input_pl_list[0]: 1.0,
            network.keep_prob_hidden_pl_list[0]: 1.0,
            network.keep_prob_output_pl_list[0]: 1.0
        }

        batch_size_each = len(inputs_seq_len)

        labels_pred_st = session.run(decode_op, feed_dict=feed_dict)
        labels_pred = sparsetensor2list(labels_pred_st, batch_size_each)
        for i_batch in range(batch_size_each):

            # Convert from list to string
            str_true = num2char(labels_true[i_batch], map_file_path)
            str_pred = num2char(labels_pred[i_batch], map_file_path)

            # Remove silence(_) labels
            str_true = re.sub(r'[_\'\":;!?,.-]+', "", str_true)
            str_pred = re.sub(r'[_\'\":;!?,.-]+', "", str_pred)

            # Convert to lower case
            if label_type == 'character_capital_divide':
                str_true = str_true.lower()
                str_pred = str_pred.lower()

            # Compute edit distance
            cer_mean += Levenshtein.distance(
                str_pred, str_true) / len(list(str_true))

        if next_epoch_flag:
            break

    cer_mean /= dataset.data_num

    return cer_mean
