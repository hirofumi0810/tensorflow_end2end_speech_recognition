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
from utils.evaluation.edit_distance import compute_per, compute_cer, compute_wer, wer_align


def do_eval_per(session, decode_op, per_op, model, dataset, label_type,
                is_test=False, eval_batch_size=None, progressbar=False,
                is_multitask=False):
    """Evaluate trained model by Phone Error Rate.
    Args:
        session: session of training model
        decode_op: operation for decoding
        per_op: operation for computing phone error rate
        model: the model to evaluate
        dataset: An instance of a `Dataset' class
        label_type (string): phone39 or phone48 or phone61
        is_test (bool, optional): set to True when evaluating by the test set
        eval_batch_size (int, optional): the batch size when evaluating the model
        progressbar (bool, optional): if True, visualize the progressbar
        is_multitask (bool, optional): if True, evaluate the multitask model
    Returns:
        per_mean (float): An average of PER
    """
    batch_size_original = dataset.batch_size

    # Reset data counter
    dataset.reset()

    # Set batch size in the evaluation
    if eval_batch_size is not None:
        dataset.batch_size = eval_batch_size

    train_label_type = label_type
    eval_label_type = dataset.label_type_sub if is_multitask else dataset.label_type

    idx2phone_train = Idx2phone(
        map_file_path='../metrics/mapping_files/' + train_label_type + '.txt')
    idx2phone_eval = Idx2phone(
        map_file_path='../metrics/mapping_files/' + eval_label_type + '.txt')
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
            model.inputs_pl_list[0]: inputs[0],
            model.inputs_seq_len_pl_list[0]: inputs_seq_len[0],
            model.keep_prob_pl_list[0]: 1.0
        }

        batch_size = inputs[0].shape[0]

        # Evaluate by 39 phones
        labels_pred_st = session.run(decode_op, feed_dict=feed_dict)
        labels_pred = sparsetensor2list(labels_pred_st, batch_size)

        for i_batch in range(batch_size):
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
            if is_test:
                phone_true_list = labels_true[0][i_batch][0].split(' ')
            else:
                # Convert from index to phone (-> list of phone strings)
                phone_true_list = idx2phone_eval(
                    labels_true[0][i_batch]).split(' ')

            # Mapping to 39 phones (-> list of phone strings)
            phone_true_list = map2phone39_eval(phone_true_list)

            # Compute PER
            per_mean += compute_per(ref=phone_pred_list,
                                    hyp=phone_true_list,
                                    normalize=True)

            if progressbar:
                pbar.update(1)

        if is_new_epoch:
            break

    per_mean /= len(dataset)

    # Register original batch size
    if eval_batch_size is not None:
        dataset.batch_size = batch_size_original

    return per_mean


def do_eval_cer(session, decode_op, model, dataset, label_type,
                is_test=False, eval_batch_size=None, progressbar=False,
                is_multitask=False):
    """Evaluate trained model by Character Error Rate.
    Args:
        session: session of training model
        decode_op: operation for decoding
        model: the model to evaluate
        dataset: An instance of a `Dataset` class
        label_type (string): character or character_capital_divide
        is_test (bool, optional): set to True when evaluating by the test set
        eval_batch_size (int, optional): the batch size when evaluating the model
        progressbar (bool, optional): if True, visualize the progressbar
        is_multitask (bool, optional): if True, evaluate the multitask model
    Return:
        cer_mean (float): An average of CER
        wer_mean (float): An average of WER
    """
    batch_size_original = dataset.batch_size

    # Reset data counter
    dataset.reset()

    # Set batch size in the evaluation
    if eval_batch_size is not None:
        dataset.batch_size = eval_batch_size

    if label_type == 'character':
        idx2char = Idx2char(
            map_file_path='../metrics/mapping_files/character.txt')
    elif label_type == 'character_capital_divide':
        idx2char = Idx2char(
            map_file_path='../metrics/mapping_files/character_capital_divide.txt',
            capital_divide=True,
            space_mark='_')

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
            model.inputs_pl_list[0]: inputs[0],
            model.inputs_seq_len_pl_list[0]: inputs_seq_len[0],
            model.keep_prob_pl_list[0]: 1.0
        }

        batch_size = inputs[0].shape[0]

        labels_pred_st = session.run(decode_op, feed_dict=feed_dict)
        labels_pred = sparsetensor2list(labels_pred_st, batch_size)
        for i_batch in range(batch_size):

            # Convert from list of index to string
            if is_test:
                str_true = labels_true[0][i_batch][0]
                # NOTE: transcript is seperated by space('_')
            else:
                # Convert from list of index to string
                str_true = idx2char(labels_true[0][i_batch],
                                    padded_value=dataset.padded_value)
            str_pred = idx2char(labels_pred[i_batch])

            # Remove consecutive spaces
            str_pred = re.sub(r'[_]+', '_', str_pred)

            # Remove garbage labels
            str_true = re.sub(r'[\'\":;!?,.-]+', '', str_true)
            str_pred = re.sub(r'[\'\":;!?,.-]+', '', str_pred)

            # Compute WER
            wer_mean += compute_wer(hyp=str_pred.split('_'),
                                    ref=str_true.split('_'),
                                    normalize=True)
            # substitute, insert, delete = wer_align(
            #     ref=str_pred.split('_'),
            #     hyp=str_true.split('_'))
            # print('SUB: %d' % substitute)
            # print('INS: %d' % insert)
            # print('DEL: %d' % delete)

            # Remove spaces
            str_pred = re.sub(r'[_]+', '', str_pred)
            str_true = re.sub(r'[_]+', '', str_true)

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

    # Register original batch size
    if eval_batch_size is not None:
        dataset.batch_size = batch_size_original

    return cer_mean, wer_mean
