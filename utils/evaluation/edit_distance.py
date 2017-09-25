#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Functions for computing edit distance."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import Levenshtein as lev


def compute_edit_distance(session, labels_true_st, labels_pred_st):
    """Compute edit distance per mini-batch.
    Args:
        session:
        labels_true_st: A `SparseTensor` of ground truth
        labels_pred_st: A `SparseTensor` of prediction
    Returns:
        edit_distances: list of edit distance of each uttearance
    """
    indices, values, dense_shape = labels_true_st
    labels_pred_pl = tf.SparseTensor(indices, values, dense_shape)
    indices, values, dense_shape = labels_pred_st
    labels_true_pl = tf.SparseTensor(indices, values, dense_shape)

    edit_op = tf.edit_distance(labels_pred_pl, labels_true_pl, normalize=True)
    edit_distances = session.run(edit_op)

    return edit_distances


def compute_cer(str_pred, str_true, normalize=True):
    """Compute Character Error Rate.
    Args:
        str_pred (string): a sentence without spaces
        str_true (string): a sentence without spaces
        normalize (bool, optional): if True, divide by the length of str_true
    Returns:
        cer (float): Character Error Rate between str_true and str_pred
    """
    cer = lev.distance(str_pred, str_true)
    if normalize:
        cer /= len(list(str_true))
    return cer


def compute_wer(str_pred, str_true, space_mark=' ', normalize=True):
    """Compute Word Error Rate.
    Args:
        str_pred (string): a space separated sentence
        str_true (string): a space separated sentence
        normalize (bool, optional): if True, divide by the length of str_true
        space_mark (string): a string to represent the space
    Returns:
        wer (float): word error rate between str_pred and str_true
    """
    words_pred = str_pred.split(space_mark)
    words_true = str_true.split(space_mark)

    # Prepare mapping from word to index
    vocab_tmp = set(words_pred + words_true)
    word2idx = dict(zip(vocab_tmp, range(len(vocab_tmp))))

    # Map words to unique characters
    seq_pred = ''.join([chr(word2idx[word]) for word in words_pred])
    seq_true = ''.join([chr(word2idx[word]) for word in words_true])
    # NOTE: Levenshtein packages only accepts strings)

    wer = lev.distance(seq_pred, seq_true)
    if normalize:
        wer /= len(words_true)
    return wer
