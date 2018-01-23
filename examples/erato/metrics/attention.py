#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Define evaluation method for the Attention-based model (ERATO corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.io.labels.character import Idx2char
from utils.evaluation.edit_distance import compute_cer


def do_eval_cer(session, decode_op, model, dataset, label_type, ss_type,
                is_test=False, eval_batch_size=None, progressbar=False):
    """Evaluate trained model by Character Error Rate.
    Args:
        session: session of training model
        decode_op: operation for decoding
        model: the model to evaluate
        dataset: An instance of a `Dataset` class
        label_type (string): kana
        ss_type (string): remove or insert_left or insert_both or insert_right
        is_test (bool, optional): set to True when evaluating by the test set
        eval_batch_size (int, optional): the batch size when evaluating the model
        progressbar (bool, optional): if True, visualize the progressbar
    Return:
        cer_mean (float): An average of CER
    """
    batch_size_original = dataset.batch_size

    # Reset data counter
    dataset.reset()

    # Set batch size in the evaluation
    if eval_batch_size is not None:
        dataset.batch_size = eval_batch_size

    idx2char = Idx2char(
        map_file_path='../metrics/mapping_files/' + label_type + '_' + ss_type + '.txt')

    cer_mean = 0
    if progressbar:
        pbar = tqdm(total=len(dataset))
    for data, is_new_epoch in dataset:

        # Create feed dictionary for next mini batch
        inputs, labels_true, inputs_seq_len, labels_seq_len,  _ = data

        feed_dict = {
            model.inputs_pl_list[0]: inputs[0],
            model.inputs_seq_len_pl_list[0]: inputs_seq_len[0],
            model.keep_prob_encoder_pl_list[0]: 1.0,
            model.keep_prob_decoder_pl_list[0]: 1.0,
            model.keep_prob_embedding_pl_list[0]: 1.0
        }

        batch_size = inputs[0].shape[0]

        labels_pred = session.run(decode_op, feed_dict=feed_dict)

        for i_batch in range(batch_size):

            # Convert from list of index to string
            if is_test:
                str_true = labels_true[0][i_batch][0]
                # NOTE: transcript is seperated by space('_')
            else:
                str_true = idx2char(
                    labels_true[0][i_batch][1:labels_seq_len[0][i_batch] - 1])
            str_pred = idx2char(labels_pred[i_batch]).split('>')[0]
            # NOTE: Trancate by <EOS>

            # Remove garbage labels
            str_true = re.sub(r'[_NZLFBDlfbdー<>]+', '', str_true)
            str_pred = re.sub(r'[_NZLFBDlfbdー<>]+', '', str_pred)

            # Compute CER
            cer_mean += compute_cer(str_pred=str_pred,
                                    str_true=str_true,
                                    normalize=True)

            if progressbar:
                pbar.update(1)

        if is_new_epoch:
            break

    cer_mean /= len(dataset)

    # Register original batch size
    if eval_batch_size is not None:
        dataset.batch_size = batch_size_original

    return cer_mean


def do_eval_fmeasure(session, decode_op, model, dataset, label_type, ss_type,
                     is_test=False, eval_batch_size=None, progressbar=False):
    """Evaluate trained model by F-measure.
    Args:
        session: session of training model
        decode_op: operation for decoding
        model: the model to evaluate
        dataset: An instance of a `Dataset` class
        label_type (string): kana
        ss_type (string): remove or insert_left or insert_both or insert_right
        is_test (bool, optional): set to True when evaluating by the test set
        eval_batch_size (int, optional): the batch size when evaluating the model
        progressbar (bool, optional): if True, visualize the progressbar
    Returns:
        f_mean (float): An average of F-measure of each social signal
    """
    batch_size_original = dataset.batch_size

    # Reset data counter
    dataset.reset()

    # Set batch size in the evaluation
    if eval_batch_size is not None:
        dataset.batch_size = eval_batch_size

    idx2char = Idx2char(
        map_file_path='../metrics/mapping_files/' + label_type + '_' + ss_type + '.txt')

    tp_lau, fp_lau, fn_lau = 0., 0., 0.
    tp_fil, fp_fil, fn_fil = 0., 0., 0.
    tp_bac, fp_bac, fn_bac = 0., 0., 0.
    tp_dis, fp_dis, fn_dis = 0., 0., 0.
    if progressbar:
        pbar = tqdm(total=len(dataset))
    for data, is_new_epoch in dataset:

        # Create feed dictionary for next mini batch
        inputs, labels_true, inputs_seq_len, labels_seq_len,  _ = data

        feed_dict = {
            model.inputs_pl_list[0]: inputs[0],
            model.inputs_seq_len_pl_list[0]: inputs_seq_len[0],
            model.keep_prob_encoder_pl_list[0]: 1.0,
            model.keep_prob_decoder_pl_list[0]: 1.0,
            model.keep_prob_embedding_pl_list[0]: 1.0
        }

        batch_size = inputs[0].shape[0]

        labels_pred = session.run(decode_op, feed_dict=feed_dict)

        for i_batch in range(batch_size):

            # Convert from list of index to string
            if is_test:
                str_true = labels_true[0][i_batch][0]
                # NOTE: transcript is seperated by space('_')
            else:
                # Convert from list of index to string
                str_true = idx2char(
                    labels_true[0][i_batch][1:labels_seq_len[0][i_batch] - 1],
                    padded_value=dataset.padded_value)
            str_pred = idx2char(labels_pred[i_batch]).split('>')[0]
            # NOTE: Trancate by <EOS>

            detected_lau_num = str_pred.count('L')
            detected_fil_num = str_pred.count('F')
            detected_bac_num = str_pred.count('B')
            detected_dis_num = str_pred.count('D')

            true_lau_num = str_true.count('L')
            true_fil_num = str_true.count('F')
            true_bac_num = str_true.count('B')
            true_dis_num = str_true.count('D')

            # Laughter
            if detected_lau_num <= true_lau_num:
                tp_lau += detected_lau_num
                fn_lau += true_lau_num - detected_lau_num
            else:
                tp_lau += true_lau_num
                fp_lau += detected_lau_num - true_lau_num

            # Filler
            if detected_fil_num <= true_fil_num:
                tp_fil += detected_fil_num
                fn_fil += true_fil_num - detected_fil_num
            else:
                tp_fil += true_fil_num
                fp_fil += detected_fil_num - true_fil_num

            # Backchannel
            if detected_bac_num <= true_bac_num:
                tp_bac += detected_bac_num
                fn_bac += true_bac_num - detected_bac_num
            else:
                tp_bac += true_bac_num
                fp_bac += detected_bac_num - true_bac_num

            # Disfluency
            if detected_dis_num <= true_dis_num:
                tp_dis += detected_dis_num
                fn_dis += true_dis_num - detected_dis_num
            else:
                tp_dis += true_dis_num
                fp_dis += detected_dis_num - true_dis_num

            if progressbar:
                pbar.update(1)

        if is_new_epoch:
            break

    p_lau = tp_lau / (tp_lau + fp_lau) if (tp_lau + fp_lau) != 0 else 0
    r_lau = tp_lau / (tp_lau + fn_lau) if (tp_lau + fn_lau) != 0 else 0
    f_lau = 2 * r_lau * p_lau / (r_lau + p_lau) if (r_lau + p_lau) != 0 else 0

    r_fil = tp_fil / (tp_fil + fn_fil) if (tp_fil + fn_fil) != 0 else 0
    p_fil = tp_fil / (tp_fil + fp_fil) if (tp_fil + fp_fil) != 0 else 0
    f_fil = 2 * r_fil * p_fil / (r_fil + p_fil) if (r_fil + p_fil) != 0 else 0

    p_bac = tp_bac / (tp_bac + fp_bac) if (tp_bac + fp_bac) != 0 else 0
    r_bac = tp_bac / (tp_bac + fn_bac) if (tp_bac + fn_bac) != 0 else 0
    f_bac = 2 * r_bac * p_bac / (r_bac + p_bac) if (r_bac + p_bac) != 0 else 0

    r_dis = tp_dis / (tp_dis + fn_dis) if (tp_dis + fn_dis) != 0 else 0
    p_dis = tp_dis / (tp_dis + fp_dis) if (tp_dis + fp_dis) != 0 else 0
    f_dis = 2 * r_dis * p_dis / (r_dis + p_dis) if (r_dis + p_dis) != 0 else 0

    acc_lau = [p_lau, r_lau, f_lau]
    acc_fil = [p_fil, r_fil, f_fil]
    acc_bac = [p_bac, r_bac, f_bac]
    acc_dis = [p_dis, r_dis, f_dis]
    mean = [(p_lau + p_fil + p_bac + p_dis) / 4., (r_lau + r_fil + r_bac + r_dis) / 4.,
            (f_lau + f_fil + f_bac + f_dis) / 4.]

    df_acc = pd.DataFrame({'Laughter': acc_lau,
                           'Filler': acc_fil,
                           'Backchannel': acc_bac,
                           'Disfluency': acc_dis,
                           'Mean': mean},
                          columns=['Laughter', 'Filler',
                                   'Backchannel', 'Disfluency', 'Mean'],
                          index=['Precision', 'Recall', 'F-measure'])

    # Register original batch size
    if eval_batch_size is not None:
        dataset.batch_size = batch_size_original

    return df_acc
