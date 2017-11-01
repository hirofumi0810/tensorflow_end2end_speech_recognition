#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Define evaluation method for CTC model (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from tqdm import tqdm

from utils.io.labels.character import Idx2char, Char2idx
from utils.io.labels.word import Idx2word
from utils.io.labels.sparsetensor import sparsetensor2list
from utils.evaluation.edit_distance import compute_cer, compute_wer, wer_align
from models.ctc.decoders.beam_search_decoder import BeamSearchDecoder


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
    else:
        raise TypeError

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
                        str_true = idx2char(labels_true[i_device][i_batch],
                                            padded_value=dataset.padded_value)
                    str_pred = idx2char(labels_pred[i_batch])

                    # Remove consecutive spaces
                    str_pred = re.sub(r'[_]+', '_', str_pred)

                    # Remove garbage labels
                    str_true = re.sub(r'[\']+', '', str_true)
                    str_pred = re.sub(r'[\']+', '', str_pred)

                    # Compute WER
                    wer_mean += compute_wer(ref=str_true.split('_'),
                                            hyp=str_pred.split('_'),
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

    # Register original batch size
    if eval_batch_size is not None:
        dataset.batch_size = batch_size_original

    return cer_mean, wer_mean


def do_eval_cer2(session, posteriors_ops, beam_width, model, dataset,
                 label_type, is_test=False, eval_batch_size=None,
                 progressbar=False, is_multitask=False):
    """Evaluate trained model by Character Error Rate.
    Args:
        session: session of training model
        posteriors_ops: list of operations for computing posteriors
        beam_width (int):
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
    assert isinstance(posteriors_ops, list), "posteriors_ops must be a list."

    batch_size_original = dataset.batch_size

    # Reset data counter
    dataset.reset()

    # Set batch size in the evaluation
    if eval_batch_size is not None:
        dataset.batch_size = eval_batch_size

    if label_type == 'character':
        idx2char = Idx2char(
            map_file_path='../metrics/mapping_files/character.txt')
        char2idx = Char2idx(
            map_file_path='../metrics/mapping_files/character.txt')
    elif label_type == 'character_capital_divide':
        raise NotImplementedError
    else:
        raise TypeError

    # Define decoder
    decoder = BeamSearchDecoder(space_index=char2idx('_')[0],
                                blank_index=model.num_classes - 1)

    cer_mean, wer_mean = 0, 0
    if progressbar:
        pbar = tqdm(total=len(dataset))
    for data, is_new_epoch in dataset:

        # Create feed dictionary for next mini batch
        if is_multitask:
            inputs, _, labels_true, inputs_seq_len, _ = data
        else:
            inputs, labels_true, inputs_seq_len, _ = data

        feed_dict = {}
        for i_device in range(len(posteriors_ops)):
            feed_dict[model.inputs_pl_list[i_device]] = inputs[i_device]
            feed_dict[model.inputs_seq_len_pl_list[i_device]
                      ] = inputs_seq_len[i_device]
            feed_dict[model.keep_prob_pl_list[i_device]] = 1.0

        posteriors_list = session.run(posteriors_ops, feed_dict=feed_dict)
        for i_device, labels_pred_st in enumerate(posteriors_list):
            batch_size_device, max_time = inputs[i_device].shape[:2]

            posteriors = posteriors_list[i_device].reshape(
                batch_size_device, max_time, model.num_classes)

            for i_batch in range(batch_size_device):

                # Decode per utterance
                labels_pred, scores = decoder(
                    probs=posteriors[i_batch:i_batch + 1],
                    seq_len=inputs_seq_len[i_device][i_batch: i_batch + 1],
                    beam_width=beam_width)

                # Convert from list of index to string
                if is_test:
                    str_true = labels_true[i_device][i_batch][0]
                    # NOTE: transcript is seperated by space('_')
                else:
                    str_true = idx2char(labels_true[i_device][i_batch],
                                        padded_value=dataset.padded_value)
                str_pred = idx2char(labels_pred[0])

                # Remove consecutive spaces
                str_pred = re.sub(r'[_]+', '_', str_pred)

                # Remove garbage labels
                str_true = re.sub(r'[\']+', '', str_true)
                str_pred = re.sub(r'[\']+', '', str_pred)

                # Compute WER
                wer_mean += compute_wer(ref=str_true.split('_'),
                                        hyp=str_pred.split('_'),
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

        if is_new_epoch:
            break

    cer_mean /= (len(dataset))
    wer_mean /= (len(dataset))

    # Register original batch size
    if eval_batch_size is not None:
        dataset.batch_size = batch_size_original

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

    batch_size_original = dataset.batch_size

    # Reset data counter
    dataset.reset()

    # Set batch size in the evaluation
    if eval_batch_size is not None:
        dataset.batch_size = eval_batch_size

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

    # Register original batch size
    if eval_batch_size is not None:
        dataset.batch_size = batch_size_original

    return wer_mean
