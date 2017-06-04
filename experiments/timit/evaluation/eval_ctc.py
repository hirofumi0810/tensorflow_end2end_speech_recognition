#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Define evaluation method for CTC network (TIMIT corpus)."""

import re
import numpy as np
import Levenshtein
from tqdm import tqdm

from plot import probs
from utils.labels.character import num2char
from utils.labels.phone import num2phone, phone2num
from .util import map_to_39phone, compute_edit_distance
from utils.data.sparsetensor import list2sparsetensor, sparsetensor2list
from utils.exception_func import exception
from utils.util import join


def do_eval_per(session, decode_op, per_op, network, dataset, label_type,
                eval_batch_size=1, rate=1.0, is_progressbar=False):
    """Evaluate trained model by Phone Error Rate.
    Args:
        session: session of training model
        decode_op: operation for decoding
        per_op: operation for computing phone error rate
        network: network to evaluate
        dataset: `Dataset' class
        label_type: phone39 or phone48 or phone61 or character
        eval_batch_size: batch size on evaluation
        rate: A float value. Rate of evaluation data to use
        is_progressbar: if True, evaluate during training, else during restoring
    Returns:
        per_global: phone error rate
    """
    if label_type not in ['phone39', 'phone48', 'phone61']:
        raise ValueError(
            'data_type is "phone39" or "phone48" or "phone61".')

    batch_size = eval_batch_size

    num_examples = dataset.data_num * rate
    iteration = int(num_examples / batch_size)
    if (num_examples / batch_size) != int(num_examples / batch_size):
        iteration += 1
    per_global = 0

    # Setting for progressbar
    iterator = tqdm(range(iteration)) if is_progressbar else range(iteration)

    p2n_map_file_path = '../evaluation/mapping_files/ctc/phone2num_' + \
        label_type[5:7] + '.txt'
    p2n39_map_file_path = '../evaluation/mapping_files/ctc/phone2num_39.txt'
    p2p_map_file_path = '../evaluation/mapping_files/phone2phone.txt'
    for step in iterator:
        # Create feed dictionary for next mini batch
        inputs, labels_true, seq_len, _ = dataset.next_batch(
            batch_size=batch_size)
        indices, values, dense_shape = list2sparsetensor(labels_true)

        feed_dict = {
            network.inputs_pl: inputs,
            network.seq_len_pl: seq_len,
            network.keep_prob_input_pl: 1.0,
            network.keep_prob_hidden_pl: 1.0
        }

        batch_size_each = len(labels_true)

        if False:
            # evaluate by 61 phones
            per_local = session.run(per_op, feed_dict=feed_dict)
            per_global += per_local * batch_size_each

        else:
            # evaluate by 39 phones
            labels_pred_st = session.run(decode_op, feed_dict=feed_dict)
            labels_pred = sparsetensor2list(labels_pred_st, batch_size_each)
            for i_batch in range(batch_size_each):
                # Convert to phone (list of phone strings)
                phone_pred_seq = num2phone(
                    labels_pred[i_batch], p2n_map_file_path)
                phone_true_seq = num2phone(
                    labels_true[i_batch], p2n_map_file_path)
                phone_pred_list = phone_pred_seq.split(' ')
                phone_true_list = phone_true_seq.split(' ')

                # Mapping to 39 phones (list of phone strings)
                phone_pred_list = map_to_39phone(
                    phone_pred_list, label_type, p2p_map_file_path)
                phone_true_list = map_to_39phone(
                    phone_true_list, label_type, p2p_map_file_path)

                # Convert to num (list of phone indices)
                phone_pred_list = phone2num(
                    phone_pred_list, p2n39_map_file_path)
                phone_true_list = phone2num(
                    phone_true_list, p2n39_map_file_path)
                labels_pred[i_batch] = phone_pred_list
                labels_true[i_batch] = phone_true_list

            # Compute edit distance
            labels_true_st = list2sparsetensor(labels_true)
            labels_pred_st = list2sparsetensor(labels_pred)
            per_local = compute_edit_distance(
                session, labels_true_st, labels_pred_st)
            per_global += per_local * batch_size_each

    per_global /= dataset.data_num
    print('  Phone Error Rate: %f %%' % (per_global * 100))

    return per_global


@exception
def do_eval_cer(session, decode_op, network, dataset,
                eval_batch_size=1, rate=1.0, is_progressbar=False):
    """Evaluate trained model by Character Error Rate.
    Args:
        session: session of training model
        decode_op: operation for decoding
        network: network to evaluate
        dataset: Dataset class
        eval_batch_size: batch size on evaluation
        rate: rate of evaluation data to use
        is_progressbar: if True, visualize progressbar
    Return:
        cer_mean: mean character error rate
    """
    batch_size = eval_batch_size

    num_examples = dataset.data_num * rate
    iteration = int(num_examples / batch_size)
    if (num_examples / batch_size) != int(num_examples / batch_size):
        iteration += 1
    cer_sum = 0

    # Setting for progressbar
    iterator = tqdm(range(iteration)) if is_progressbar else range(iteration)

    map_file_path = '../evaluation/mapping_files/ctc/char2num.txt'
    for step in iterator:
        # Create feed dictionary for next mini batch
        inputs, labels, seq_len, _ = dataset.next_batch(batch_size=batch_size)
        indices, values, dense_shape = list2sparsetensor(labels)

        feed_dict = {
            network.inputs_pl: inputs,
            network.seq_len_pl: seq_len,
            network.keep_prob_input_pl: 1.0,
            network.keep_prob_hidden_pl: 1.0
        }

        batch_size_each = len(labels)
        labels_st = session.run(decode_op, feed_dict=feed_dict)
        labels_pred = sparsetensor2list(labels_st, batch_size_each)
        for i_batch in range(batch_size_each):

            # Convert from list to string
            str_pred = num2char(labels_pred[i_batch], map_file_path)
            str_true = num2char(labels[i_batch], map_file_path)

            # Remove silence(_) labels
            str_pred = re.sub(r'[_]+', "", str_pred)
            str_true = re.sub(r'[_]+', "", str_true)

            # Compute edit distance
            cer_each = Levenshtein.distance(
                str_pred, str_true) / len(list(str_true))
            cer_sum += cer_each

    cer_mean = cer_sum / dataset.data_num
    print('  Character Error Rate: %f %%' % (cer_mean * 100))
    return cer_mean


def decode_test(session, decode_op, network, dataset, label_type, rate=1.0):
    """Visualize label outputs.
    Args:
        session: session of training model
        decode_op: operation for decoding
        network: network to evaluate
        dataset: Dataset class
        label_type: phone39 or phone48 or phone61 or character
        rate: rate of evaluation data to use
    """
    batch_size = 1
    num_examples = dataset.data_num * rate
    iteration = int(num_examples / batch_size)
    if (num_examples / batch_size) != int(num_examples / batch_size):
        iteration += 1

    map_file_path_phone = '../evaluation/mapping_files/ctc/phone2num_' + \
        label_type[5:7] + '.txt'
    map_file_path_char = '../evaluation/mapping_files/ctc/char2num.txt'
    for step in range(iteration):
        # Create feed dictionary for next mini batch
        inputs, labels, seq_len, input_names = dataset.next_batch(
            batch_size=batch_size)
        indices, values, dense_shape = list2sparsetensor(labels)

        feed_dict = {
            network.inputs_pl: inputs,
            network.seq_len_pl: seq_len,
            network.keep_prob_input_pl: 1.0,
            network.keep_prob_hidden_pl: 1.0
        }

        # Visualize
        batch_size_each = len(labels)
        labels_st = session.run(decode_op, feed_dict=feed_dict)
        labels_pred = sparsetensor2list(labels_st, batch_size_each)
        for i_batch in range(batch_size_each):
            if label_type == 'character':
                print('-----wav: %s-----' % input_names[i_batch])
                print('True: %s' % num2char(
                    labels[i_batch], map_file_path_char))
                print('Pred: %s' % num2char(
                    labels_pred[i_batch], map_file_path_char))

            else:
                # Decode test (39 phones)
                print('-----wav: %s-----' % input_names[i_batch])
                print('True: %s' % num2phone(
                    labels[i_batch], map_file_path_phone))

                print('Pred: %s' % num2phone(
                    labels_pred[i_batch], map_file_path_phone))


def posterior_test(session, posteriors_op, network, dataset, label_type, rate=1.0):
    """Visualize label posteriors.
    Args:
        session: session of training model
        posteriois_op: operation for computing posteriors
        network: network to evaluate
        dataset: Dataset class
        label_type: phone39 or phone48 or phone61 or character
        rate: rate of evaluation data to use
    """
    save_path = join(network.model_dir, 'ctc_output')
    batch_size = 1
    num_examples = dataset.data_num * rate
    iteration = int(num_examples / batch_size)
    if (num_examples / batch_size) != int(num_examples / batch_size):
        iteration += 1

    for step in range(iteration):
        # Create feed dictionary for next mini batch
        inputs, labels, seq_len, input_names = dataset.next_batch(
            batch_size=batch_size)
        indices, values, dense_shape = list2sparsetensor(labels)

        feed_dict = {
            network.inputs_pl: inputs,
            network.seq_len_pl: seq_len,
            network.keep_prob_input_pl: 1.0,
            network.keep_prob_hidden_pl: 1.0
        }

        # Visualize
        batch_size_each = len(labels)
        max_frame_num = inputs.shape[1]
        posteriors = session.run(posteriors_op, feed_dict=feed_dict)
        for i_batch in range(batch_size_each):
            posteriors_index = np.array([i_batch + (batch_size_each * j)
                                         for j in range(max_frame_num)])
            if label_type[:5] == 'phone':
                probs.plot_probs_ctc_phone(probs=posteriors[posteriors_index][:int(seq_len[i_batch]), :],
                                           save_path=save_path,
                                           wav_index=input_names[i_batch],
                                           data_type=dataset.data_type,
                                           label_type=label_type)
            # else:
            #     probs.plot_probs_ctc_char(probs=posteriors[posteriors_index][:int(seq_len[i_batch]), :],
            #                               save_path=save_path,
            #                               wav_index=input_names[i_batch],
            #                               data_type=dataset.data_type)
