#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""define evaluation method for frame-wise classifiers (SVC corpus)."""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('../')
from . import metric
from plot import probs


def do_eval_uaauc(session, posteriors_op, network, dataset, rate=1.0, is_training=True):
    """Evaluate trained model by UAAUC.
    Args:
        session: session of trained model
        posteriors_op: operation for computing posteriors
        network: network to evaluate
        dataset: set of input data and labels
        rate: rate of data to use
        is_training: if True, evaluate during training, else during restoring
    Returns:
        auc_l: AUC of laughter
        auc_f: AUC of siller
        uaauc: UAAUC between laughter and filler
    """
    num_examples = dataset.data_num * rate
    iteration = max(1, int(num_examples / network.batch_size) + 1)
    iter_laughter = iteration
    iter_filler = iteration
    auc_l_sum = 0
    auc_f_sum = 0

    # setting for progressbar
    if is_training:
        iterator = range(iteration)
    else:
        iterator = tqdm(range(iteration))

    for step in iterator:
        # create feed dictionary for next mini batch
        inputs, labels = dataset.next_batch(batch_size=network.batch_size)

        feed_dict = {
            network.inputs_pl: inputs,
            network.labels_pl: labels
        }

        for i in range(len(network.keep_prob_pl_list)):
            feed_dict[network.keep_prob_pl_list[i]] = 1.0

        posteriors = session.run(posteriors_op, feed_dict=feed_dict)

        # low pass filter

        # logistic regression

        # check if positive label is included in mini batch
        labels_tmp = np.copy(labels)
        auc_l_batch = metric.compute_auc(labels, posteriors, label_index=1)
        auc_f_batch = metric.compute_auc(labels_tmp, posteriors, label_index=2)

        # nan check
        if auc_l_batch != auc_l_batch:
            iter_laughter -= 1
        else:
            auc_l_sum += auc_l_batch
        if auc_f_batch != auc_f_batch:
            iter_filler -= 1
        else:
            auc_f_sum += auc_f_batch

    auc_l = auc_l_sum / iter_laughter
    auc_f = auc_f_sum / iter_filler
    uaauc = (auc_l + auc_f) / 2.

    acc_l = [auc_l, uaauc]
    acc_f = [auc_f, uaauc]

    df_auc = pd.DataFrame({'Laughter': acc_l, 'Filler': acc_f},
                          columns=['Laughter', 'Filler'],
                          index=['AUC', 'UAAUC'])

    print(df_auc)

    return auc_l, auc_f, uaauc


def do_eval_fmeasure(session, decode_op, network, dataset, rate=1.0, is_training=True):
    """Evaluate trained model by F-measure.
    Args:
        session: session of training model
        decode_op: operation for decoding
        network: network to evaluate
        dataset: `Dataset' class
        rate: A float value. Rate of evaluation data to use
        is_training: if True, evaluate during training, else during restoring
    Returns:
        acc_l: list of [precision, recall, f_measure] of laughter
        acc_f: list of [precision, recall, f_measure] of filler
        fmean: mean of f-measure between laughter and filler
    """
    num_examples = dataset.data_num * rate
    iteration = max(1, int(num_examples / network.batch_size) + 1)
    tp_l, fp_l, fn_l = 0, 0, 0
    tp_f, fp_f, fn_f = 0, 0, 0

    iter_laughter = iteration
    iter_filler = iteration

    # setting for progressbar
    if is_training:
        iterator = range(iteration)
    else:
        p = ProgressBar(max_value=iteration)
        iterator = p(range(iteration))

    for step in iterator:
        # create feed dictionary for next mini batch
        inputs, labels = dataset.next_batch(batch_size=network.batch_size)

        feed_dict = {
            network.inputs_pl: inputs,
            network.labels_pl: labels
        }

        batch_size = len(labels)

        for i in range(len(network.keep_prob_pl_list)):
            feed_dict[network.keep_prob_pl_list[i]] = 1.0

        posteriors = session.run(posteriors_op, feed_dict=feed_dict)

        # HMM processing

        for i_batch in range(batch_size):
            detected_l_num = np.sum(np.array(labels_pred[i_batch]) == 1)
            detected_f_num = np.sum(np.array(labels_pred[i_batch]) == 2)
            true_l_num = np.sum(labels[i_batch] == 1)
            true_f_num = np.sum(labels[i_batch] == 2)

            # laughter
            if detected_l_num <= true_l_num:
                tp_l += detected_l_num
                fn_l += true_l_num - detected_l_num
            else:
                tp_l += true_l_num
                fp_l += detected_l_num - true_l_num

            # filler
            if detected_f_num <= true_f_num:
                tp_f += detected_f_num
                fn_f += true_f_num - detected_f_num
            else:
                tp_f += true_f_num
                fp_f += detected_f_num - true_f_num

    p_l = tp_l / (tp_l + fp_l) if (tp_l + fp_l) != 0 else 0
    r_l = tp_l / (tp_l + fn_l) if (tp_l + fn_l) != 0 else 0
    f_l = 2 * r_l * p_l / (r_l + p_l) if (r_l + p_l) != 0 else 0

    r_f = tp_f / (tp_f + fn_f) if (tp_f + fn_f) != 0 else 0
    p_f = tp_f / (tp_f + fp_f) if (tp_f + fp_f) != 0 else 0
    f_f = 2 * r_f * p_f / (r_f + p_f) if (r_f + p_f) != 0 else 0

    confusion_l = [tp_l, fp_l, fn_l, tp_l + fp_l + fn_l]
    confusion_f = [tp_f, fp_f, fn_f, tp_f + fp_f + fn_f]
    acc_l = [p_l, r_l, f_l]
    acc_f = [p_f, r_f, f_f]
    mean = [(p_l + p_f) / 2., (r_l + r_f) / 2., (f_l + f_f) / 2.]

    df_confusion = pd.DataFrame({'Laughter': confusion_l, 'Filler': confusion_f},
                                columns=['Laughter', 'Filler'],
                                index=['TP', 'FP', 'FN', 'Sum'])
    df_acc = pd.DataFrame({'Laughter': acc_l, 'Filler': acc_f, 'Mean': mean},
                          columns=['Laughter', 'Filler', 'Mean'],
                          index=['Precision', 'Recall', 'F-measure'])

    # print(df_confusion)
    print(df_acc)

    return acc_l, acc_f, mean[2]
