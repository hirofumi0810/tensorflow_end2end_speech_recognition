#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def list2sparsetensor(labels):
    """Convert labels from list to sparse tensor.
    Args:
        labels: list of labels
    Returns:
        labels_st: sparse tensor of labels, list of indices, values, dense_shape
    """
    indices, values = [], []
    for i_utt, each_label in enumerate(labels):
        for i_l, l in enumerate(each_label):
            indices.append([i_utt, i_l])
            values.append(l)
    dense_shape = [len(labels), np.asarray(indices).max(0)[1] + 1]
    labels_st = [np.array(indices), np.array(values), np.array(dense_shape)]

    return labels_st


def sparsetensor2list(labels_st, batch_size):
    """Convert labels from sparse tensor to list.
    Args:
        labels_st: sparse tensor of labels
    Returns:
        labels: list of labels
    """
    indices = labels_st.indices
    values = labels_st.values

    labels = []
    batch_boundary = np.where(indices[:, 1] == 0)[0]

    # print(batch_boundary)
    # if len(batch_boundary) != batch_size:
    #     batch_boundary = np.array(batch_boundary.tolist() + [max(batch_boundary) + 1])
    # print(indices)

    for i in range(batch_size - 1):
        label_each_wav = values[batch_boundary[i]:batch_boundary[i + 1]]
        labels.append(label_each_wav.tolist())
    labels.append(values[batch_boundary[-1]:].tolist())

    return labels
