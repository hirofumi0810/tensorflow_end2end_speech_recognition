#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilities for RNN layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def sequence_length(inputs, time_major=True, dtype=tf.int32):
    """Inspect the length of the sequence of input data.
    Args:
        inputs: A tensor of size `[B, T, input_size]`
        time_major (bool, optional): set True if inputs is time-major
        dtype (optional): default is tf.int32
    Returns:
        seq_len: A tensor of size `[B,]`
    """
    time_axis = 0 if time_major else 1
    with tf.variable_scope("seq_len"):
        used = tf.sign(tf.reduce_max(tf.abs(inputs), axis=2))
        seq_len = tf.reduce_sum(used, axis=time_axis)
        seq_len = tf.cast(seq_len, dtype)
    return seq_len
