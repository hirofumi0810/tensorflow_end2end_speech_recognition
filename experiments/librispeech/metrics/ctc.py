#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Define evaluation method for CTC model (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from tqdm import tqdm

from utils.io.labels.character import Idx2char
from utils.io.labels.sparsetensor import sparsetensor2list
from utils.evaluation.edit_distance import compute_cer, compute_wer, wer_align


def do_eval_cer(session, decode_ops, model, dataset, label_type,
                eval_batch_size=None, progressbar=False, is_multitask=False):
    """Evaluate trained model by Character Error Rate.
    Args:
        session: session of training model
        decode_ops: list of operations for decoding
        model: the model to evaluate
        dataset: An instance of a `Dataset` class
        label_type (string): character or character_capital_divide
        eval_batch_size (int, optional): int, the batch size when evaluating the model
        progressbar (bool, optional): if True, visualize the progressbar
        is_multitask (bool, optional): if True, evaluate the multitask model
    Return:
        cer_mean (float): An average of CER
        wer_mean (float): An average of WER
    """
    assert isinstance(decode_ops, list), "decode_ops must be a list."

    # Reset data counter
    dataset.reset()

    if label_type == 'character':
        idx2char = Idx2char(
            map_file_path='../metrics/mapping_files/ctc/character.txt')
    elif label_type == 'character_capital_divide':
        idx2char = Idx2char(
            map_file_path='../metrics/mapping_files/ctc/character_capital_divide.txt',
            capital_divide=True,
            space_mark='_')

    cer_mean, wer_mean = 0, 0
    skip_data_num = 0
    if progressbar:
        pbar = tqdm(total=len(dataset))
    for data, is_new_epoch in dataset:

        # Create feed dictionary for next mini batch
        if is_multitask:
            inputs, _, labels_true, inputs_seq_len, _ = data
        else:
            inputs, labels_true, inputs_seq_len, _ = data

        feed_dict = {}
        for i_device in range(len(decode_ops)):
            feed_dict[model.inputs_pl_list[i_device]] = inputs[i_device]
            feed_dict[model.inputs_seq_len_pl_list[i_device]
                      ] = inputs_seq_len[i_device]
            feed_dict[model.keep_prob_input_pl_list[i_device]] = 1.0
            feed_dict[model.keep_prob_hidden_pl_list[i_device]] = 1.0
            feed_dict[model.keep_prob_output_pl_list[i_device]] = 1.0

        labels_pred_st_list = session.run(decode_ops, feed_dict=feed_dict)
        for i_device, labels_pred_st in enumerate(labels_pred_st_list):
            batch_size_device = len(inputs_seq_len[i_device])
            try:
                labels_pred = sparsetensor2list(labels_pred_st,
                                                batch_size_device)
                for i_batch in range(batch_size_device):

                    # Convert from list of index to string
                    str_true = idx2char(labels_true[i_device][i_batch])
                    str_pred = idx2char(labels_pred[i_batch])

                    # Remove consecutive spaces
                    str_pred = re.sub(r'[_]+', '_', str_pred)

                    # Remove garbage labels
                    str_true = re.sub(r'[\']+', "", str_true)
                    str_pred = re.sub(r'[\']+', "", str_pred)

                    # Compute WER
                    wer_mean += compute_wer(ref=str_pred.split('_'),
                                            hyp=str_true.split('_'),
                                            normalize=True)
                    # substitute, insert, delete = wer_align(
                    #     ref=str_pred.split('_'),
                    #     hyp=str_true.split('_'))
                    # print(substitute)
                    # print(insert)
                    # print(delete)

                    # Remove spaces
                    str_true = re.sub(r'[_]+', "", str_true)
                    str_pred = re.sub(r'[_]+', "", str_pred)

                    # Compute CER
                    cer_mean += compute_cer(str_pred=str_pred,
                                            str_true=str_true,
                                            normalize=True)

                    if progressbar:
                        pbar.update(1)

            except IndexError:
                print('skipped')
                skip_data_num += batch_size_device
                # TODO: Conduct decoding again with batch size 1

                if progressbar:
                    pbar.update(batch_size_device)

        if is_new_epoch:
            break

    cer_mean /= (len(dataset) - skip_data_num)
    wer_mean /= (len(dataset) - skip_data_num)
    # TODO: Fix this

    return cer_mean, wer_mean


def do_eval_wer(session, decode_ops, model, dataset, train_data_size,
                is_test=False, eval_batch_size=None, progressbar=False,
                is_multitask=False):
    """Evaluate trained model by Word Error Rate.
    Args:
        session: session of training model
        decode_ops: list of operations for decoding
        model: the model to evaluate
        dataset: An instance of `Dataset` class
        train_data_size (string): train_clean100 or train_clean360 or
            train_other500 or train_all
        is_test (bool): set to True when evaluating by the test set
        eval_batch_size (int, optional): the batch size when evaluating the model
        progressbar (bool, optional): if True, visualize progressbar
        is_multitask (bool, optional): if True, evaluate the multitask model
    Return:
        wer_mean (bool): An average of WER
    """
    assert isinstance(decode_ops, list), "decode_ops must be a list."

    # Reset data counter
    dataset.reset()

    wer_mean = 0
    skip_data_num = 0
    if progressbar:
        pbar = tqdm(total=len(dataset))
    for data, is_new_epoch in dataset:

        # Create feed dictionary for next mini batch
        if is_multitask:
            inputs, labels_true, _, inputs_seq_len, _ = data
        else:
            inputs, labels_true, inputs_seq_len, _ = data

        feed_dict = {}
        for i_device in range(len(decode_ops)):
            feed_dict[model.inputs_pl_list[i_device]] = inputs[i_device]
            feed_dict[model.inputs_seq_len_pl_list[i_device]
                      ] = inputs_seq_len[i_device]
            feed_dict[model.keep_prob_input_pl_list[i_device]] = 1.0
            feed_dict[model.keep_prob_hidden_pl_list[i_device]] = 1.0
            feed_dict[model.keep_prob_output_pl_list[i_device]] = 1.0

        labels_pred_st_list = session.run(decode_ops, feed_dict=feed_dict)
        for i_device, labels_pred_st in enumerate(labels_pred_st_list):
            batch_size_device = len(inputs_seq_len[i_device])
            try:
                labels_pred = sparsetensor2list(labels_pred_st,
                                                batch_size_device)

                for i_batch in range(batch_size_device):

                    # Compute WER
                    wer_mean += compute_wer(ref=labels_pred[i_batch],
                                            hyp=labels_true[i_device][i_batch],
                                            normalize=True)
                    # substitute, insert, delete = wer_align(
                    #     ref=labels_pred[i_batch],
                    #     hyp=labels_true[i_device][i_batch])
                    # print(substitute)
                    # print(insert)
                    # print(delete)

                    if progressbar:
                        pbar.update(1)

            except IndexError:
                print('skipped')
                skip_data_num += batch_size_device
                # TODO: Conduct decoding again with batch size 1

                if progressbar:
                    pbar.update(batch_size_device)

        if is_new_epoch:
            break

    wer_mean /= (len(dataset) - skip_data_num)
    # TODO: This is just edit distance.

    return wer_mean
