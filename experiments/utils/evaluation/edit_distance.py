#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Functions for computing edit distance."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


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
