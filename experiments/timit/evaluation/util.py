#! /usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf


def map_to_39phone(phone_list, label_type, map_file_path):
    """Map from 61 or 48 phones to 39 phones.
    Args:
        phone_list: list of phones (string)
        label_type: phone48 or phone61
        map_file_path: path to the mapping file
    Returns:
        phone_list: list of 39 phones (string)
    """
    if label_type == 'phone39':
        return phone_list

    # read a mapping file
    map_dict = {}
    with open(map_file_path) as f:
        for line in f:
            line = line.strip().split()
            if label_type == 'phone61':
                if line[1] != 'nan':
                    map_dict[line[0]] = line[2]
                else:
                    map_dict[line[0]] = ''
            elif label_type == 'phone48':
                if line[1] != 'nan':
                    map_dict[line[1]] = line[2]

    # map to 39 phones
    for i in range(len(phone_list)):
        phone_list[i] = map_dict[phone_list[i]]

    # ignore q (only if 61 phones)
    while '' in phone_list:
        phone_list.remove('')

    return phone_list


def compute_edit_distance(session, labels_true_st, labels_pred_st):
    """Compute edit distance.
    Args:
        session:
        labels_true_st: A `SparseTensor` of ground truth
        labels_pred_st: A `SparseTensor` of prediction
    Returns:
        edit_distance: edit distance
    """
    indices, values, dense_shape = labels_true_st
    labels_pred_pl = tf.SparseTensor(indices, values, dense_shape)
    indices, values, dense_shape = labels_pred_st
    labels_true_pl = tf.SparseTensor(indices, values, dense_shape)
    edit_op = tf.reduce_mean(tf.edit_distance(labels_pred_pl, labels_true_pl, normalize=True))
    edit_distance = session.run(edit_op)

    return edit_distance
