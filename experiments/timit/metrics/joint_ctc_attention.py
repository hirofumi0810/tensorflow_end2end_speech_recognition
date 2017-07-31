#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Define evaluation method for the joint CTC-Attention model (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import Levenshtein

from experiments.timit.metrics.mapping import map_to_39phone
from experiments.timit.metrics.edit_distance import compute_edit_distance
from experiments.utils.data.labels.character import num2char
from experiments.utils.data.labels.phone import num2phone, phone2num
from experiments.utils.data.sparsetensor import list2sparsetensor
from experiments.utils.progressbar import wrap_generator


def do_eval_per(session, decode_op, per_op, network, dataset, label_type,
                eos_index, eval_batch_size=None, progressbar=False):
    """Evaluate trained model by Phone Error Rate.
    Args:
        session: session of training model
        decode_op: operation for decoding
        per_op: operation for computing phone error rate
        network: network to evaluate
        dataset: An instance of a `Dataset' class
        label_type: string, phone39 or phone48 or phone61
        eos_index: int, the index of <EOS> class
        eval_batch_size: int, the batch size when evaluating the model
        progressbar: if True, visualize the progressbar
    Returns:
        per_mean: An average of PER
    """
    batch_size = dataset.batch_size if eval_batch_size is None else eval_batch_size
    train_label_type = label_type
    eval_label_type = dataset.label_type

    train_phone2num_map_file_path = '../metrics/mapping_files/attention/' + \
        train_label_type + '_to_num.txt'
    eval_phone2num_map_file_path = '../metrics/mapping_files/attention/' + \
        eval_label_type + '_to_num.txt'
    phone2num_39_map_file_path = '../metrics/mapping_files/attention/phone39_to_num.txt'
    phone2phone_map_file_path = '../metrics/mapping_files/phone2phone.txt'
    per_mean = 0
    total_step = int(dataset.data_num / batch_size)
    if (dataset.data_num / batch_size) != int(dataset.data_num / batch_size):
        total_step += 1
    for data, next_epoch_flag in wrap_generator(dataset(batch_size),
                                                progressbar,
                                                total=total_step):
        # Create feed dictionary for next mini-batch
        inputs, att_labels_true, _, inputs_seq_len, _, _ = data

        feed_dict = {
            network.inputs_pl_list[0]: inputs,
            network.inputs_seq_len_pl_list[0]: inputs_seq_len,
            network.keep_prob_input_pl_list[0]: 1.0,
            network.keep_prob_hidden_pl_list[0]: 1.0,
            network.keep_prob_output_pl_list[0]: 1.0
        }

        batch_size_each = len(inputs_seq_len)

        # Evaluate by 39 phones
        att_labels_pred = session.run(decode_op, feed_dict=feed_dict)
        # NOTE: prediction will be made from the attention outputs

        att_labels_pred_mapped, att_labels_true_mapped = [], []
        for i_batch in range(batch_size_each):
            ###############
            # Hypothesis
            ###############
            # Convert from num to phone (-> list of phone strings)
            phone_pred_list = num2phone(
                att_labels_pred[i_batch],
                train_phone2num_map_file_path).split(' ')

            # Mapping to 39 phones (-> list of phone strings)
            phone_pred_list = map_to_39phone(phone_pred_list,
                                             train_label_type,
                                             phone2phone_map_file_path)

            # Convert from phone to num (-> list of phone indices)
            phone_pred_list = phone2num(phone_pred_list,
                                        phone2num_39_map_file_path)
            att_labels_pred_mapped.append(phone_pred_list)

            ###############
            # Reference
            ###############
            # Convert from num to phone (-> list of phone strings)
            phone_true_list = num2phone(
                att_labels_true[i_batch],
                eval_phone2num_map_file_path).split(' ')

            # Mapping to 39 phones (-> list of phone strings)
            phone_true_list = map_to_39phone(phone_true_list,
                                             eval_label_type,
                                             phone2phone_map_file_path)

            # Convert from phone to num (-> list of phone indices)
            phone_true_list = phone2num(phone_true_list,
                                        phone2num_39_map_file_path)
            att_labels_true_mapped.append(phone_true_list)

        # Compute edit distance
        labels_true_st = list2sparsetensor(
            att_labels_true_mapped, padded_value=eos_index)
        labels_pred_st = list2sparsetensor(
            att_labels_pred_mapped, padded_value=eos_index)
        per_each = compute_edit_distance(session,
                                         labels_true_st,
                                         labels_pred_st)
        per_mean += per_each * batch_size_each

        if next_epoch_flag:
            break

    per_mean /= dataset.data_num

    return per_mean


def do_eval_cer(session, decode_op, network, dataset, label_type,
                eval_batch_size=None, progressbar=False):
    """Evaluate trained model by Character Error Rate.
    Args:
        session: session of training model
        decode_op: operation for decoding
        network: network to evaluate
        dataset: An instance of a `Dataset` class
        label_type: string, character or character_capital_divide
        eval_batch_size: int, batch size when evaluating the model
        progressbar: if True, visualize the progressbar
    Return:
        cer_mean: An average of CER
    """
    batch_size = dataset.batch_size if eval_batch_size is None else eval_batch_size

    map_file_path = '../metrics/mapping_files/attention/' + label_type + '_to_num.txt'
    cer_mean = 0
    total_step = int(dataset.data_num / batch_size)
    if (dataset.data_num / batch_size) != int(dataset.data_num / batch_size):
        total_step += 1
    for data, next_epoch_flag in wrap_generator(dataset(batch_size),
                                                progressbar,
                                                total=total_step):
        # Create feed dictionary for next mini-batch
        inputs, att_labels_true, _, inputs_seq_len, _, _ = data

        feed_dict = {
            network.inputs_pl_list[0]: inputs,
            network.inputs_seq_len_pl_list[0]: inputs_seq_len,
            network.keep_prob_input_pl_list[0]: 1.0,
            network.keep_prob_hidden_pl_list[0]: 1.0,
            network.keep_prob_output_pl_list[0]: 1.0
        }

        batch_size_each = len(inputs_seq_len)

        att_labels_pred = session.run(decode_op, feed_dict=feed_dict)
        # NOTE: prediction will be made from the attention outputs

        for i_batch in range(batch_size_each):

            # Convert from list to string
            str_true = num2char(att_labels_true[i_batch], map_file_path)
            str_pred = num2char(att_labels_pred[i_batch], map_file_path)

            # Remove silence(_) labels
            str_true = re.sub(r'[<>_\'\":;!?,.-]+', "", str_true)
            str_pred = re.sub(r'[<>_\'\":;!?,.-]+', "", str_pred)

            # Compute edit distance
            cer_each = Levenshtein.distance(
                str_pred, str_true) / len(list(str_true))
            cer_mean += cer_each

        if next_epoch_flag:
            break

    cer_mean /= dataset.data_num

    return cer_mean
