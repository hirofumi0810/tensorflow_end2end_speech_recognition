#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Define evaluation method for CTC model (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from tqdm import tqdm

from utils.io.labels.character import Idx2char
from utils.io.labels.word import Idx2word
from utils.io.labels.sparsetensor import sparsetensor2list
from utils.evaluation.edit_distance import compute_cer, compute_wer, wer_align


def do_eval_cer(session, decode_ops, model, dataset, label_type,
                is_test=False, eval_batch_size=None, progressbar=False,
                is_multitask=False):
    """Evaluate trained model by Character Error Rate.
    Args:
        session: session of training model
        decode_ops: list of operations for decoding
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
    assert isinstance(decode_ops, list), "decode_ops must be a list."

    # Reset data counter
    dataset.reset()

    if label_type == 'character':
        idx2char = Idx2char(
            map_file_path='../metrics/mapping_files/character.txt')
    elif label_type == 'character_capital_divide':
        idx2char = Idx2char(
            map_file_path='../metrics/mapping_files/character_capital_divide.txt',
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
            feed_dict[model.keep_prob_pl_list[i_device]] = 1.0

        labels_pred_st_list = session.run(decode_ops, feed_dict=feed_dict)
        for i_device, labels_pred_st in enumerate(labels_pred_st_list):
            batch_size_device = len(inputs[i_device])
            try:
                labels_pred = sparsetensor2list(labels_pred_st,
                                                batch_size_device)
                for i_batch in range(batch_size_device):

                    # Convert from list of index to string
                    if is_test:
                        str_true = labels_true[i_device][i_batch][0]
                        # NOTE: transcript is seperated by space('_')
                    else:
                        str_true = idx2char(labels_true[i_device][i_batch])
                    str_pred = idx2char(labels_pred[i_batch])

                    # Remove consecutive spaces
                    str_pred = re.sub(r'[_]+', '_', str_pred)

                    # Remove garbage labels
                    str_true = re.sub(r'[\']+', '', str_true)
                    str_pred = re.sub(r'[\']+', '', str_pred)

                    # Compute WER
                    wer_mean += compute_wer(ref=str_pred.split('_'),
                                            hyp=str_true.split('_'),
                                            normalize=True)
                    # substitute, insert, delete = wer_align(
                    #     ref=str_pred.split('_'),
                    #     hyp=str_true.split('_'))
                    # print('SUB: %d' % substitute)
                    # print('INS: %d' % insert)
                    # print('DEL: %d' % delete)

                    # Remove spaces
                    str_true = re.sub(r'[_]+', '', str_true)
                    str_pred = re.sub(r'[_]+', '', str_pred)

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
        train_data_size (string): train100h or train460h or train960h
        is_test (bool, optional): set to True when evaluating by the test set
        eval_batch_size (int, optional): the batch size when evaluating the model
        progressbar (bool, optional): if True, visualize progressbar
        is_multitask (bool, optional): if True, evaluate the multitask model
    Return:
        wer_mean (bool): An average of WER
    """
    assert isinstance(decode_ops, list), "decode_ops must be a list."

    # Reset data counter
    dataset.reset()

    idx2word = Idx2word(
        map_file_path='../metrics/mapping_files/word_' + train_data_size + '.txt')

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
            feed_dict[model.keep_prob_pl_list[i_device]] = 1.0

        labels_pred_st_list = session.run(decode_ops, feed_dict=feed_dict)
        for i_device, labels_pred_st in enumerate(labels_pred_st_list):
            batch_size_device = len(inputs[i_device])
            try:
                labels_pred = sparsetensor2list(labels_pred_st,
                                                batch_size_device)

                for i_batch in range(batch_size_device):

                    if is_test:
                        str_true = labels_true[i_device][i_batch][0]
                        # NOTE: transcript is seperated by space('_')
                    else:
                        str_true = '_'.join(
                            idx2word(labels_true[i_device][i_batch]))
                    str_pred = '_'.join(idx2word(labels_pred[i_batch]))

                    # if len(str_true.split('_')) == 0:
                    #     print(str_true)
                    #     print(str_pred)

                    # Compute WER
                    wer_mean += compute_wer(ref=str_true.split('_'),
                                            hyp=str_pred.split('_'),
                                            normalize=True)
                    # substitute, insert, delete = wer_align(
                    #     ref=str_true.split(' '),
                    #     hyp=str_pred.split(' '))
                    # print('SUB: %d' % substitute)
                    # print('INS: %d' % insert)
                    # print('DEL: %d' % delete)

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

    return wer_mean
