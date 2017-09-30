#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilities for decoding of the CTC model (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import sys

from utils.io.labels.character import Idx2char
from utils.io.labels.phone import Idx2phone
from utils.io.labels.sparsetensor import sparsetensor2list
from utils.evaluation.edit_distance import wer_align


def decode_test(session, decode_op, model, dataset, label_type, save_path=None):
    """Visualize label outputs of CTC model.
    Args:
        session: session of training model
        decode_op: operation for decoding
        model: the model to evaluate
        dataset: An instance of a `Dataset` class
        label_type (string): phone39 or phone48 or phone61 or character or
            character_capital_divide
        save_path (string, optional): path to save decoding results
    """
    if label_type == 'character':
        map_fn = Idx2char(
            map_file_path='../metrics/mapping_files/ctc/character.txt')
    elif label_type == 'character_capital_divide':
        map_fn = Idx2char(
            map_file_path='../metrics/mapping_files/ctc/character_capital_divide.txt',
            capital_divide=True)
    else:
        map_fn = Idx2phone(
            map_file_path='../metrics/mapping_files/ctc/' + label_type + '.txt')

    if save_path is not None:
        sys.stdout = open(join(model.model_dir, 'decode.txt'), 'w')

    while True:

        # Create feed dictionary for next mini batch
        data, is_new_epoch = dataset.next(batch_size=1)
        inputs, labels_true, inputs_seq_len, input_names = data
        # NOTE: Batch size is expected to be 1

        feed_dict = {
            model.inputs_pl_list[0]: inputs,
            model.inputs_seq_len_pl_list[0]: inputs_seq_len,
            model.keep_prob_input_pl_list[0]: 1.0,
            model.keep_prob_hidden_pl_list[0]: 1.0,
            model.keep_prob_output_pl_list[0]: 1.0
        }

        # Visualize
        labels_pred_st = session.run(decode_op, feed_dict=feed_dict)
        try:
            labels_pred = sparsetensor2list(labels_pred_st, batch_size=1)
        except IndexError:
            # no output
            labels_pred = ['']
        finally:
            print('----- wav: %s -----' % input_names[0])
            if label_type == 'character':
                true_seq = map_fn(labels_true[0]).replace('_', ' ')
                pred_seq = map_fn(labels_pred[0]).replace('_', ' ')
            else:
                true_seq = map_fn(labels_true[0])
                pred_seq = map_fn(labels_pred[0])
            print('Ref: %s' % true_seq)
            print('Hyp: %s' % pred_seq)

        if is_new_epoch:
            break


def decode_test_multitask(session, decode_op_main, decode_op_sub, model,
                          dataset, label_type_main, label_type_sub,
                          save_path=None):
    """Visualize label outputs of Multi-task CTC model.
    Args:
        session: session of training model
        decode_op_main: operation for decoding in the main task
        decode_op_sub: operation for decoding in the sub task
        model: the model to evaluate
        dataset: An instance of a `Dataset` class
        label_type_main (string): character or character_capital_divide
        label_type_sub (string): phone39 or phone48 or phone61
        save_path (string, optional): path to save decoding results
    """
    # TODO: fix

    if save_path is not None:
        sys.stdout = open(join(model.model_dir, 'decode.txt'), 'w')

    # Decode character
    print('===== ' + label_type_main + ' =====')
    idx2char = Idx2char(
        map_file_path='../metrics/mapping_files/ctc/' + label_type_main + '.txt')
    while True:

        # Create feed dictionary for next mini batch
        data, is_new_epoch = dataset.next(batch_size=1)
        inputs, labels_true, _, inputs_seq_len, input_names = data
        # NOTE: Batch size is expected to be 1

        feed_dict = {
            model.inputs_pl_list[0]: inputs,
            model.inputs_seq_len_pl_list[0]: inputs_seq_len,
            model.keep_prob_input_pl_list[0]: 1.0,
            model.keep_prob_hidden_pl_list[0]: 1.0,
            model.keep_prob_output_pl_list[0]: 1.0
        }

        # Visualize
        labels_pred_st = session.run(decode_op_main, feed_dict=feed_dict)
        labels_pred = sparsetensor2list(labels_pred_st, batch_size=1)

        print('----- wav: %s -----' % input_names[0])
        print('Ref: %s' % idx2char(labels_true[0]))
        print('Hyp: %s' % idx2char(labels_pred[0]))

        if is_new_epoch:
            break

    # Decode phone
    print('\n===== ' + label_type_sub + ' =====')
    idx2phone = Idx2phone(
        map_file_path='../metrics/mapping_files/ctc/' + label_type_sub + '.txt')
    while True:

        # Create feed dictionary for next mini batch
        data, is_new_epoch = dataset.next(batch_size=1)
        inputs, _, labels_true, inputs_seq_len, input_names = data

        feed_dict = {
            model.inputs_pl_list[0]: inputs,
            model.inputs_seq_len_pl_list[0]: inputs_seq_len,
            model.keep_prob_input_pl_list[0]: 1.0,
            model.keep_prob_hidden_pl_list[0]: 1.0,
            model.keep_prob_output_pl_list[0]: 1.0
        }

        # Visualize
        labels_pred_st = session.run(decode_op_sub, feed_dict=feed_dict)
        try:
            labels_pred = sparsetensor2list(labels_pred_st, batch_size=1)
        except IndexError:
            # no output
            labels_pred = ['']
        finally:
            print('----- wav: %s -----' % input_names[0])
            print('Ref: %s' % idx2phone(labels_true[0]))
            print('Hyp: %s' % idx2phone(labels_pred[0]))

        if is_new_epoch:
            break
