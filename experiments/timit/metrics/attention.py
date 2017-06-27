#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Define evaluation method for the Attention model (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import Levenshtein

from .mapping import map_to_39phone
from .edit_distance import compute_edit_distance
from utils.labels.character import num2char
from utils.labels.phone import num2phone, phone2num
from utils.sparsetensor import list2sparsetensor
from utils.exception_func import exception
from utils.progressbar import wrap_iterator


@exception
def do_eval_per(session, decode_op, per_op, network, dataset, param,
                eval_batch_size=None, is_progressbar=False, is_multitask=False):
    """Evaluate trained model by Phone Error Rate.
    Args:
        session: session of training model
        decode_op: operation for decoding
        per_op: operation for computing phone error rate
        network: network to evaluate
        dataset: An instance of a `Dataset' class
        param: A dictionary of parameters
        eval_batch_size: int, the batch size when evaluating the model
        is_progressbar: if True, visualize the progressbar
        is_multitask: if True, evaluate the multitask model
    Returns:
        per_global: An average of PER
    """
    if eval_batch_size is not None:
        batch_size = eval_batch_size
    else:
        batch_size = dataset.batch_size

    train_label_type = param['label_type']
    data_label_type = dataset.label_type
    eos_index = param['eos_index']

    num_examples = dataset.data_num
    iteration = int(num_examples / batch_size)
    if (num_examples / batch_size) != int(num_examples / batch_size):
        iteration += 1
    per_global = 0

    # Make data generator
    mini_batch = dataset.next_batch(batch_size=batch_size)

    phone2num_map_file_path = '../metrics/mapping_files/attention/phone2num_' + \
        train_label_type[5:7] + '.txt'
    phone2num_39_map_file_path = '../metrics/mapping_files/attention/phone2num_39.txt'
    phone2phone_map_file_path = '../metrics/mapping_files/phone2phone.txt'
    for step in wrap_iterator(range(iteration), is_progressbar):
        # Create feed dictionary for next mini-batch
        if not is_multitask:
            inputs, labels_true, inputs_seq_len, _, _ = mini_batch.__next__()
        else:
            inputs, _, labels_true, inputs_seq_len, _, _ = mini_batch.__next__()

        feed_dict = {
            network.inputs: inputs,
            network.inputs_seq_len: inputs_seq_len,
            network.keep_prob_input: 1.0,
            network.keep_prob_hidden: 1.0
        }

        batch_size_each = len(inputs_seq_len)

        if False:
            # Evaluate by 61 phones
            per_local = session.run(per_op, feed_dict=feed_dict)
            per_global += per_local * batch_size_each

        else:
            # Evaluate by 39 phones
            predicted_ids = session.run(decode_op, feed_dict=feed_dict)
            predicted_ids_phone39 = []
            labels_true_phone39 = []
            for i_batch in range(batch_size_each):
                # Convert from num to phone (-> list of phone strings)
                phone_pred_seq = num2phone(
                    predicted_ids[i_batch], phone2num_map_file_path)
                phone_pred_list = phone_pred_seq.split(' ')

                # Mapping to 39 phones (-> list of phone strings)
                phone_pred_list = map_to_39phone(
                    phone_pred_list, train_label_type,
                    phone2phone_map_file_path)

                # Convert from phone to num (-> list of phone indices)
                phone_pred_list = phone2num(
                    phone_pred_list, phone2num_39_map_file_path)
                predicted_ids_phone39.append(phone_pred_list)

                if data_label_type != 'phone39':
                    # Convert from num to phone (-> list of phone strings)
                    phone_true_seq = num2phone(
                        labels_true[i_batch], phone2num_map_file_path)
                    phone_true_list = phone_true_seq.split(' ')

                    # Mapping to 39 phones (-> list of phone strings)
                    phone_true_list = map_to_39phone(
                        phone_true_list, train_label_type,
                        phone2phone_map_file_path)

                    # Convert from phone to num (-> list of phone indices)
                    phone_true_list = phone2num(
                        phone_true_list, phone2num_39_map_file_path)
                    labels_true_phone39.append(phone_true_list)
                else:
                    labels_true_phone39 = labels_true

            # Compute edit distance
            labels_true_st = list2sparsetensor(
                labels_true_phone39, padded_value=eos_index)
            labels_pred_st = list2sparsetensor(
                predicted_ids_phone39, padded_value=eos_index)
            per_local = compute_edit_distance(
                session, labels_true_st, labels_pred_st)
            per_global += per_local * batch_size_each

    per_global /= dataset.data_num

    return per_global


@exception
def do_eval_cer(session, decode_op, network, dataset, eval_batch_size=None,
                is_progressbar=False, is_multitask=False):
    """Evaluate trained model by Character Error Rate.
    Args:
        session: session of training model
        decode_op: operation for decoding
        network: network to evaluate
        dataset: An instance of a `Dataset` class
        eval_batch_size: int, batch size when evaluating the model
        is_progressbar: if True, visualize the progressbar
        is_multitask: if True, evaluate the multitask model
    Return:
        cer_mean: An average of CER
    """
    if eval_batch_size is not None:
        batch_size = eval_batch_size
    else:
        batch_size = dataset.batch_size

    # Make data generator
    mini_batch = dataset.next_batch(batch_size=batch_size)

    num_examples = dataset.data_num
    iteration = int(num_examples / batch_size)
    if (num_examples / batch_size) != int(num_examples / batch_size):
        iteration += 1
    cer_sum = 0

    map_file_path = '../metrics/mapping_files/attention/char2num.txt'
    for step in wrap_iterator(range(iteration), is_progressbar):
        # Create feed dictionary for next mini-batch
        if not is_multitask:
            inputs, labels_true, inputs_seq_len, _, _ = mini_batch.__next__()
        else:
            inputs, labels_true, _, inputs_seq_len, _, _ = mini_batch.__next__()

        feed_dict = {
            network.inputs: inputs,
            network.inputs_seq_len: inputs_seq_len,
            network.keep_prob_input: 1.0,
            network.keep_prob_hidden: 1.0
        }

        batch_size_each = len(inputs_seq_len)

        predicted_ids = session.run(decode_op, feed_dict=feed_dict)
        for i_batch in range(batch_size_each):

            # Convert from list to string
            str_true = num2char(labels_true[i_batch], map_file_path)
            str_pred = num2char(predicted_ids[i_batch], map_file_path)

            # Remove silence(_) labels
            str_true = re.sub(r'[_<>,.\'-?!]+', "", str_true)
            str_pred = re.sub(r'[_<>,.\'-?!]+', "", str_pred)

            # Compute edit distance
            cer_each = Levenshtein.distance(
                str_pred, str_true) / len(list(str_true))
            cer_sum += cer_each

    cer_mean = cer_sum / dataset.data_num

    return cer_mean
