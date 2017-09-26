#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Define evaluation method for the CTC model (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from tqdm import tqdm

from experiments.timit.metrics.mapping import Map2phone39
from utils.io.labels.character import Idx2char
from utils.io.labels.phone import Idx2phone
from utils.io.labels.sparsetensor import sparsetensor2list
from utils.evaluation.edit_distance import compute_cer, compute_wer


def do_eval_per(session, decode_op, per_op, model, dataset, label_type,
                eval_batch_size=None, progressbar=False, is_multitask=False):
    """Evaluate trained model by Phone Error Rate.
    Args:
        session: session of training model
        decode_op: operation for decoding
        per_op: operation for computing phone error rate
        model: the model to evaluate
        dataset: An instance of a `Dataset' class
        label_type (string): phone39 or phone48 or phone61
        eval_batch_size (int, optional): the batch size when evaluating the model
        progressbar (bool, optional): if True, visualize the progressbar
        is_multitask (bool, optional): if True, evaluate the multitask model
    Returns:
        per_mean (float): An average of PER
    """
    # Reset data counter
    dataset.reset()

    train_label_type = label_type
    eval_label_type = dataset.label_type_sub if is_multitask else dataset.label_type

    # phone2idx_39_map_file_path = '../metrics/mapping_files/ctc/phone39.txt'
    idx2phone_train = Idx2phone(
        map_file_path='../metrics/mapping_files/ctc/' + train_label_type + '.txt')
    idx2phone_eval = Idx2phone(
        map_file_path='../metrics/mapping_files/ctc/' + eval_label_type + '.txt')
    map2phone39_train = Map2phone39(
        label_type=train_label_type,
        map_file_path='../metrics/mapping_files/phone2phone.txt')
    map2phone39_eval = Map2phone39(
        label_type=eval_label_type,
        map_file_path='../metrics/mapping_files/phone2phone.txt')

    per_mean = 0
    if progressbar:
        pbar = tqdm(total=len(dataset))
    for data, is_new_epoch in dataset:

        # Create feed dictionary for next mini batch
        if is_multitask:
            inputs, _, labels_true, inputs_seq_len, _ = data
        else:
            inputs, labels_true, inputs_seq_len, _ = data

        feed_dict = {
            model.inputs_pl_list[0]: inputs,
            model.inputs_seq_len_pl_list[0]: inputs_seq_len,
            model.keep_prob_input_pl_list[0]: 1.0,
            model.keep_prob_hidden_pl_list[0]: 1.0,
            model.keep_prob_output_pl_list[0]: 1.0
        }

        batch_size_each = len(inputs_seq_len)

        # Evaluate by 39 phones
        labels_pred_st = session.run(decode_op, feed_dict=feed_dict)
        labels_pred = sparsetensor2list(labels_pred_st, batch_size_each)

        for i_batch in range(batch_size_each):
            ###############
            # Hypothesis
            ###############
            # Convert from index to phone (-> list of phone strings)
            phone_pred_list = idx2phone_train(labels_pred[i_batch]).split(' ')

            # Mapping to 39 phones (-> list of phone strings)
            phone_pred_list = map2phone39_train(phone_pred_list)

            ###############
            # Reference
            ###############
            # Convert from index to phone (-> list of phone strings)
            phone_true_list = idx2phone_eval(labels_true[i_batch]).split(' ')

            # Mapping to 39 phones (-> list of phone strings)
            phone_true_list = map2phone39_eval(phone_true_list)

            # Compute PER
            per_mean += compute_wer(str_pred=' '.join(phone_pred_list),
                                    str_true=' '.join(phone_true_list),
                                    normalize=True,
                                    space_mark=' ')

            if progressbar:
                pbar.update(1)

        if is_new_epoch:
            break

    per_mean /= len(dataset)

    return per_mean


def do_eval_cer(session, decode_op, model, dataset, label_type,
                eval_batch_size=None, progressbar=False, is_multitask=False):
    """Evaluate trained model by Character Error Rate.
    Args:
        session: session of training model
        decode_op: operation for decoding
        model: the model to evaluate
        dataset: An instance of a `Dataset` class
        label_type (string): character or character_capital_divide
        eval_batch_size (int, optional): the batch size when evaluating the model
        progressbar (bool, optional): if True, visualize the progressbar
        is_multitask (bool, optional): if True, evaluate the multitask model
    Return:
        cer_mean (float): An average of CER
        wer_mean (float): An average of WER
    """
    # Reset data counter
    dataset.reset()

    idx2char = Idx2char(
        map_file_path='../metrics/mapping_files/ctc/' + label_type + '.txt')

    cer_mean, wer_mean = 0, 0
    if progressbar:
        pbar = tqdm(total=len(dataset))
    for data, is_new_epoch in dataset:

        # Create feed dictionary for next mini batch
        if is_multitask:
            inputs, labels_true, _, inputs_seq_len, _ = data
        else:
            inputs, labels_true, inputs_seq_len, _ = data

        feed_dict = {
            model.inputs_pl_list[0]: inputs,
            model.inputs_seq_len_pl_list[0]: inputs_seq_len,
            model.keep_prob_input_pl_list[0]: 1.0,
            model.keep_prob_hidden_pl_list[0]: 1.0,
            model.keep_prob_output_pl_list[0]: 1.0
        }

        batch_size_each = len(inputs_seq_len)

        labels_pred_st = session.run(decode_op, feed_dict=feed_dict)
        labels_pred = sparsetensor2list(labels_pred_st, batch_size_each)
        for i_batch in range(batch_size_each):

            # Convert from list of index to string
            str_true = idx2char(labels_true[i_batch])
            str_pred = idx2char(labels_pred[i_batch])

            # Remove consecutive spaces
            str_pred = re.sub(r'[_]+', '_', str_pred)

            # Remove garbage labels
            str_true = re.sub(r'[\'\":;!?,.-]+', "", str_true)
            str_pred = re.sub(r'[\'\":;!?,.-]+', "", str_pred)

            # Convert to lower case
            if label_type == 'character_capital_divide':
                str_true = str_true.lower()
                str_pred = str_pred.lower()

            # Compute WER
            wer_mean += compute_wer(str_pred=str_pred,
                                    str_true=str_true,
                                    normalize=True,
                                    space_mark='_')

            # Remove spaces
            str_pred = re.sub(r'[_]+', '_', str_pred)
            str_true = re.sub(r'[_]+', '_', str_true)

            # Compute CER
            cer_mean += compute_cer(str_pred=str_pred,
                                    str_true=str_true,
                                    normalize=True)

            if progressbar:
                pbar.update(1)

        if is_new_epoch:
            break

    cer_mean /= len(dataset)
    wer_mean /= len(dataset)

    return cer_mean, wer_mean
