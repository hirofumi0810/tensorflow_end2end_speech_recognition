#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def list2sparsetensor(labels, padded_value):
    """Convert labels from list to sparse tensor.
    Args:
        labels: list of labels
        padded_value: int, the value used for padding
    Returns:
        labels_st: A SparseTensor of labels,
            list of indices, values, dense_shape
    """
    indices, values = [], []
    for i_utt, each_label in enumerate(labels):
        for i_l, l in enumerate(each_label):
            # -1 means empty
            # Plaese see details in
            # some_timit/data/read_dataset_ctc.py
            if l == padded_value:
                break
            indices.append([i_utt, i_l])
            values.append(l)
    dense_shape = [len(labels), np.asarray(indices).max(0)[1] + 1]
    labels_st = [np.array(indices, dtype=np.int64),
                 np.array(values, dtype=np.int32),
                 np.array(dense_shape, dtype=np.int64)]

    return labels_st


def sparsetensor2list(labels_st, batch_size):
    """Convert labels from sparse tensor to list.
    Args:
        labels_st: A SparseTensor of labels
        batch_size: int the size of mini-batch
    Returns:
        labels: list of labels
    """
    if isinstance(labels_st, tf.SparseTensorValue):
        indices = labels_st.indices
        values = labels_st.values
    else:
        # expected to list of [indices, values, shape]
        indices = labels_st[0]
        values = labels_st[1]

    labels = []
    batch_boundary = np.where(indices[:, 1] == 0)[0]

    # TODO: fix error occurred when ctc models do not output any labels
    # print(batch_boundary)
    # if len(batch_boundary) != batch_size:
    #     batch_boundary = np.array(batch_boundary.tolist() + [max(batch_boundary) + 1])
    # print(indices)

    for i in range(batch_size - 1):
        label_each_wav = values[batch_boundary[i]:batch_boundary[i + 1]]
        labels.append(label_each_wav.tolist())
    labels.append(values[batch_boundary[-1]:].tolist())

    return labels
