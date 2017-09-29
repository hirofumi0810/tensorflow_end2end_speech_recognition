#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilities for decoding of the CTC model (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import sys

from utils.io.labels.character import Idx2char
from utils.io.labels.word import Idx2word
from utils.io.labels.sparsetensor import sparsetensor2list
from utils.evaluation.edit_distance import wer_align


def decode_test(session, decode_op, model, dataset, label_type,
                train_data_size, save_path=None):
    """Visualize label outputs of CTC model.
    Args:
        session: session of training model
        decode_op: operation for decoding
        model: the model to evaluate
        dataset: An instance of a `Dataset` class
        label_type (string):  character or character_capital_divide or word
        train_data_size (string, optional): train_clean100 or train_clean360 or
            train_other500 or train_all
        save_path (string, optional): path to save decoding results
    """
    if label_type == 'character':
        idx2char = Idx2char(
            map_file_path='../metrics/mapping_files/ctc/character.txt')
    elif label_type == 'character_capital_divide':
        idx2char = Idx2char(
            map_file_path='../metrics/mapping_files/ctc/character_capital_divide.txt',
            capital_divide=True)
    elif label_type == 'word':
        idx2word = Idx2word(
            map_file_path='../metrics/mapping_files/ctc/word_' + train_data_size + '.txt')

    if save_path is not None:
        sys.stdout = open(join(model.model_dir, 'decode.txt'), 'w')

    while True:

        # Create feed dictionary for next mini batch
        data, is_new_epoch = dataset.next(batch_size=1)
        inputs, labels_true, inputs_seq_len, input_names = data
        # NOTE: Batch size is expected to be 1

        feed_dict = {
            model.inputs_pl_list[0]: inputs[0],
            model.inputs_seq_len_pl_list[0]: inputs_seq_len[0],
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
            print('----- wav: %s -----' % input_names[0][0])
            if label_type == 'character':
                true_seq = idx2char(labels_true[0][0]).replace('_', ' ')
                pred_seq = idx2char(labels_pred[0]).replace('_', ' ')
            elif label_type == 'character_capital_divide':
                true_seq = idx2char(labels_true[0][0])
                pred_seq = idx2char(labels_pred[0])
            else:
                if dataset.is_test:
                    true_seq = labels_true[0][0][0]
                else:
                    true_seq = ' '.join(idx2word(labels_true[0][0]))
                pred_seq = ' '.join(idx2word(labels_pred[0]))

            print('Ref: %s' % true_seq)
            print('Hyp: %s' % pred_seq)
            # wer_align(ref=true_seq.split(), hyp=pred_seq.split())

        if is_new_epoch:
            break


def decode_test_multitask(session, decode_op_main, decode_op_sub, model,
                          dataset, train_data_size, label_type_main,
                          label_type_sub, is_test=False, save_path=None):
    """Visualize label outputs of Multi-task CTC model.
    Args:
        session: session of training model
        decode_op_main: operation for decoding in the main task
        decode_op_sub: operation for decoding in the sub task
        model: the model to evaluate
        dataset: An instance of a `Dataset` class
        label_type_main (string): word
        label_type_sub (string): character or character_capital_divide
        train_data_size (string, optional): train_clean100 or train_clean360 or
            train_other500 or train_all
        save_path (string, optional): path to save decoding results
    """
    idx2word = Idx2word(
        map_file_path='../metrics/mapping_files/ctc/word_' + train_data_size + '.txt')
    idx2char = Idx2char(
        map_file_path='../metrics/mapping_files/ctc/' + label_type_sub + '.txt')

    if save_path is not None:
        sys.stdout = open(join(model.model_dir, 'decode.txt'), 'w')

    while True:

        # Create feed dictionary for next mini batch
        data, is_new_epoch = dataset.next(batch_size=1)
        inputs, labels_true_word, labels_true_char, inputs_seq_len, input_names = data
        # NOTE: Batch size is expected to be 1

        feed_dict = {
            model.inputs_pl_list[0]: inputs[0],
            model.inputs_seq_len_pl_list[0]: inputs_seq_len[0],
            model.keep_prob_input_pl_list[0]: 1.0,
            model.keep_prob_hidden_pl_list[0]: 1.0,
            model.keep_prob_output_pl_list[0]: 1.0
        }

        # Visualize
        labels_pred_st_word, labels_pred_st_char = session.run(
            [decode_op_main, decode_op_sub], feed_dict=feed_dict)
        try:
            labels_pred_word = sparsetensor2list(
                labels_pred_st_word, batch_size=1)
        except IndexError:
            # no output
            labels_pred_word = ['']

        try:
            labels_pred_char = sparsetensor2list(
                labels_pred_st_char, batch_size=1)
        except IndexError:
            # no output
            labels_pred_char = ['']

        print('----- wav: %s -----' % input_names[0][0])
        if dataset.is_test:
            true_word_seq = labels_true_word[0][0][0]
        else:
            true_word_seq = ' '.join(idx2word(labels_true_word[0][0]))
        pred_word_seq = ' '.join(idx2word(labels_pred_word[0]))
        print('Ref (word): %s' % true_word_seq)
        print('Hyp (word): %s' % pred_word_seq)

        true_char_seq = idx2char(labels_true_char[0][0])
        pred_char_seq = idx2char(labels_pred_char[0]).replace('_', ' ')
        print('Ref (char): %s' % true_char_seq)
        print('Hyp (char): %s' % pred_char_seq)

        if is_new_epoch:
            break
