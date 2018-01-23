#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Define evaluation method for CTC model (SVC corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.io.labels.sparsetensor import sparsetensor2list


def do_eval_fmeasure(session, decode_op, model, dataset,
                     eval_batch_size=None, progressbar=False):
    """Evaluate trained model by F-measure.
    Args:
        session: session of training model
        decode_op: operation for decoding
        model: the model to evaluate
        dataset: An instance of a `Dataset' class
        label_type (string): phone39 or phone48 or phone61
        is_test (bool, optional): set to True when evaluating by the test set
        eval_batch_size (int, optional): the batch size when evaluating the model
        progressbar (bool, optional): if True, visualize the progressbar
    Return:
        fmean (float): mean of f-measure of laughter and filler
    """
    # Reset data counter
    dataset.reset()

    if eval_batch_size is None:
        batch_size = dataset.batch_size
    else:
        batch_size = eval_batch_size

    tp_l, fp_l, fn_l = 0, 0, 0
    tp_f, fp_f, fn_f = 0, 0, 0
    if progressbar:
        pbar = tqdm(total=len(dataset))
    for data, is_new_epoch in dataset:

        # Create feed dictionary for next mini batch
        inputs, labels_true, inputs_seq_len, _ = data
        feed_dict = {
            model.inputs_pl_list[0]: inputs[0],
            model.inputs_seq_len_pl_list[0]: inputs_seq_len[0],
            model.keep_prob_pl_list[0]: 1.0
        }

        batch_size = inputs[0].shape[0]

        # Decode
        labels_pred_st = session.run(decode_op, feed_dict=feed_dict)
        labels_pred = sparsetensor2list(labels_pred_st, batch_size)

        for i_batch in range(batch_size):

            detected_l_num = np.sum(np.array(labels_pred[i_batch]) == 1)
            detected_f_num = np.sum(np.array(labels_pred[i_batch]) == 2)
            true_l_num = np.sum(labels_true[0][i_batch] == 1)
            true_f_num = np.sum(labels_true[0][i_batch] == 2)

            # Laughter
            if detected_l_num <= true_l_num:
                tp_l += detected_l_num
                fn_l += true_l_num - detected_l_num
            else:
                tp_l += true_l_num
                fp_l += detected_l_num - true_l_num

            # Filler
            if detected_f_num <= true_f_num:
                tp_f += detected_f_num
                fn_f += true_f_num - detected_f_num
            else:
                tp_f += true_f_num
                fp_f += detected_f_num - true_f_num

            if progressbar:
                pbar.update(1)

        if is_new_epoch:
            break

    # Compute F-measure
    p_l = tp_l / (tp_l + fp_l) if (tp_l + fp_l) != 0 else 0
    r_l = tp_l / (tp_l + fn_l) if (tp_l + fn_l) != 0 else 0
    f_l = 2 * r_l * p_l / (r_l + p_l) if (r_l + p_l) != 0 else 0

    r_f = tp_f / (tp_f + fn_f) if (tp_f + fn_f) != 0 else 0
    p_f = tp_f / (tp_f + fp_f) if (tp_f + fp_f) != 0 else 0
    f_f = 2 * r_f * p_f / (r_f + p_f) if (r_f + p_f) != 0 else 0

    # confusion_l = [tp_l, fp_l, fn_l, tp_l + fp_l + fn_l]
    # confusion_f = [tp_f, fp_f, fn_f, tp_f + fp_f + fn_f]
    acc_l = [p_l, r_l, f_l]
    acc_f = [p_f, r_f, f_f]
    mean = [(p_l + p_f) / 2., (r_l + r_f) / 2., (f_l + f_f) / 2.]

    # df_confusion = pd.DataFrame({'Laughter': confusion_l, 'Filler': confusion_f},
    #                             columns=['Laughter', 'Filler'],
    #                             index=['TP', 'FP', 'FN', 'Sum'])
    # print(df_confusion)

    df_acc = pd.DataFrame({'Laughter': acc_l, 'Filler': acc_f, 'Mean': mean},
                          columns=['Laughter', 'Filler', 'Mean'],
                          index=['Precision', 'Recall', 'F-measure'])
    # print(df_acc)

    return mean[2], df_acc


def do_eval_fmeasure_time(session, decode_op, posteriors_op, model, dataset,
                          eval_batch_size=None, progressbar=False):
    """Evaluate trained model by F-measure.
    Args:
        session: session of training model
        decode_op: operation for decoding
        posteriors_op: operation for computing posteriors
        model: the model to evaluate
        dataset: An instance of a `Dataset' class
        label_type (string): phone39 or phone48 or phone61
        is_test (bool, optional): set to True when evaluating by the test set
        eval_batch_size (int, optional): the batch size when evaluating the model
        progressbar (bool, optional): if True, visualize the progressbar
    Returns:
        fmean (float): mean of f-measure of laughter and filler
    """
    threshold_l = threshold_f = 0.5

    # Load ground truth labels
    utterance_dict = read_trans(
        label_path='/n/sd8/inaguma/corpus/svc/data/labels.txt')

    batch_size_original = dataset.batch_size

    # Reset data counter
    dataset.reset()

    # Set batch size in the evaluation
    if eval_batch_size is not None:
        dataset.batch_size = eval_batch_size

    tp_l, fp_l, fn_l = 0, 0, 0
    tp_f, fp_f, fn_f = 0, 0, 0
    if progressbar:
        pbar = tqdm(total=len(dataset))
    for data, is_new_epoch in dataset:

        # Create feed dictionary for next mini batch
        inputs, labels_true, inputs_seq_len, input_names = data
        feed_dict = {
            model.inputs_pl_list[0]: inputs[0],
            model.inputs_seq_len_pl_list[0]: inputs_seq_len[0],
            model.keep_prob_pl_list[0]: 1.0
        }

        batch_size = inputs[0].shape[0]

        max_frame_num = inputs.shape[1]
        posteriors = session.run([posteriors_op], feed_dict=feed_dict)
        for i_batch in range(batch_size):

            # posteriors of each class
            posteriors_index = np.array([i_batch + (batch_size * j)
                                         for j in range(max_frame_num)])
            posteriors_each = posteriors[posteriors_index]
            posteriors_l = posteriors_each[:, 1]
            posteriors_f = posteriors_each[:, 2]
            predict_frames_l = np.where(posteriors_l >= threshold_l)[0]
            predict_frames_f = np.where(posteriors_f >= threshold_f)[0]

            # summarize consecutive frames in each spike
            predict_frames_l_summary = []
            predict_frames_f_summary = []
            for i_frame in range(len(predict_frames_l)):
                # not last frame
                if i_frame != len(predict_frames_l) - 1:
                    # not consecutive
                    if predict_frames_l[i_frame] + 1 != predict_frames_l[i_frame + 1]:
                        predict_frames_l_summary.append(
                            predict_frames_l[i_frame])
                else:
                    predict_frames_l_summary.append(predict_frames_l[i_frame])
            for i_frame in range(len(predict_frames_f)):
                # not last frame
                if i_frame != len(predict_frames_f) - 1:
                    # not consecutive
                    if predict_frames_f[i_frame] + 1 != predict_frames_f[i_frame + 1]:
                        predict_frames_f_summary.append(
                            predict_frames_f[i_frame])
                else:
                    predict_frames_f_summary.append(predict_frames_f[i_frame])

            # compute true interval of each class
            utt_info_list = utterance_dict[input_names[i_batch]]
            true_frames_l = np.zeros((max_frame_num,))
            true_frames_f = np.zeros((max_frame_num,))
            for i_label in range(len(utt_info_list)):
                start_frame = utt_info_list[i_label][1]
                end_frame = utt_info_list[i_label][2]
                if utt_info_list[i_label][0] == 'laughter':
                    true_frames_l[start_frame:end_frame] = 1
                elif utt_info_list[i_label][0] == 'filler':
                    true_frames_f[start_frame:end_frame] = 1

            detect_l_num = len(predict_frames_l_summary)
            detect_f_num = len(predict_frames_f_summary)
            true_l_num = np.sum(labels_true[i_batch] == 1)
            true_f_num = np.sum(labels_true[i_batch] == 2)

            ####################
            # laughter
            ####################
            for frame in predict_frames_l_summary:
                # prediction is true
                if true_frames_l[frame] == 1:
                    # TODO: まだ予測してない
                    tp_l += 1
                    # TODO: すでに予測してたら無視
                else:
                    fp_l += 1
            # could not predict
            if true_l_num > detect_l_num:
                fn_l += true_l_num - detect_l_num

            ####################
            # filler
            ####################
            for frame in predict_frames_f_summary:
                # prediction is true
                if true_frames_f[frame] == 1:
                    # TODO: まだ予測してない
                    tp_f += 1
                    # TODO: すでに予測してたら無視
                else:
                    fp_f += 1
            # could not predict
            if true_f_num > detect_f_num:
                fn_f += true_f_num - detect_f_num

            if progressbar:
                pbar.update(1)

    p_l = tp_l / (tp_l + fp_l) if (tp_l + fp_l) != 0 else 0
    r_l = tp_l / (tp_l + fn_l) if (tp_l + fn_l) != 0 else 0
    f_l = 2 * r_l * p_l / (r_l + p_l) if (r_l + p_l) != 0 else 0

    r_f = tp_f / (tp_f + fn_f) if (tp_f + fn_f) != 0 else 0
    p_f = tp_f / (tp_f + fp_f) if (tp_f + fp_f) != 0 else 0
    f_f = 2 * r_f * p_f / (r_f + p_f) if (r_f + p_f) != 0 else 0

    # confusion_l = [tp_l, fp_l, fn_l, tp_l + fp_l + fn_l]
    # confusion_f = [tp_f, fp_f, fn_f, tp_f + fp_f + fn_f]
    acc_l = [p_l, r_l, f_l]
    acc_f = [p_f, r_f, f_f]
    mean = [(p_l + p_f) / 2., (r_l + r_f) / 2., (f_l + f_f) / 2.]

    # df_confusion = pd.DataFrame({'Laughter': confusion_l, 'Filler': confusion_f},
    #                             columns=['Laughter', 'Filler'],
    #                             index=['TP', 'FP', 'FN', 'Sum'])
    # print(df_confusion)

    df_acc = pd.DataFrame({'Laughter': acc_l, 'Filler': acc_f, 'Mean': mean},
                          columns=['Laughter', 'Filler', 'Mean'],
                          index=['Precision', 'Recall', 'F-measure'])
    # print(df_acc)

    # Register original batch size
    if eval_batch_size is not None:
        dataset.batch_size = batch_size_original

    return mean[2], df_acc


def read_trans(label_path):
    """Read labels.
    Args:
        label_path (list): list of paths to target labels
    Returns:
        utterance_dict (dict):
            key (string) => speaker + '_' + utt_index
            value (list) => [label, start_frame, end_frame, label, ...]
    """
    utterance_dict = {}
    with open(label_path, 'r') as f:
        for line in f.readlines()[1:]:

            # Extract utterance information
            line = line.strip('\n').split(',')
            utt_index = line[0]
            speaker = line[1]
            # gender = line[2]
            # global_start_time = line[3]
            line = line[4:]

            # Extract laughter and filler
            labels_utt = []
            for i in range(int(len(line) / 3)):
                label = line[0]
                start = float(line[1])
                end = float(line[2])

                labels_utt.append(
                    [label, int(round(start * 100)), int(round(end * 100))])

                line = line[3:]

            utterance_dict[speaker + '_' + utt_index] = labels_utt

    return utterance_dict


def do_eval_ler(session, ler_op, model, dataset, progressbar=False):
    """Evaluate trained model by Label Error Rate.
    Args:
        session: session of training model
        ler_op: operation for computing label error rate
        model: the model to evaluate
        dataset: An instance of a `Dataset` class
        progressbar (bool, optional): if True, visualize the progressbar
    Returns:
        ler_mean (float): An average of LER
    """
    ler_mean = 0
    if progressbar:
        pbar = tqdm(total=len(dataset))
    for data, is_new_epoch in dataset:

        # create feed dictionary for next mini batch
        inputs, labels_true, inputs_seq_len, _ = data
        feed_dict = {
            model.inputs_pl_list[0]: inputs[0],
            model.inputs_seq_len_pl_list[0]: inputs_seq_len[0],
            model.keep_prob_pl_list[0]: 1.0
        }

        batch_size = inputs[0].shape[0]

        ler_batch = session.run(ler_op, feed_dict=feed_dict)
        ler_mean += ler_batch * batch_size

        if progressbar:
            pbar.update(batch_size)

    ler_mean /= dataset.data_num

    return ler_mean
