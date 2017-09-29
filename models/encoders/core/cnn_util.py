#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilities for CNN-like layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def max_pool(bottom, name='max_pool'):
    """A max pooling layer.
    Args:
        bottom: A tensor of size `[B * T, H, W, C]`
        name (string): A layer name
    Returns:
        A tensor of size `[B * T, H / 2, W / 2, C]`
    """
    return tf.nn.max_pool(
        bottom,
        ksize=[1, 2, 2, 1],  # original
        # ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding='SAME', name=name)


def avg_pool(bottom, name='avg_pool'):
    """An average pooling layer.
    Args:
        bottom: A tensor of size `[B * T, H, W, C]`
        name (string): A layer name
    Returns:
        A tensor of size `[B * T, H / 2, W / 2, C]`
    """
    return tf.nn.avg_pool(
        bottom,
        ksize=[1, 2, 2, 1],  # original
        # ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
        name=name)


def conv_layer(bottom, filter_shape, parameter_init,
               relu=True, name='conv'):
    """A convolutional layer
    Args:
        bottom: A tensor of size `[B * T, H, W, C]`
        filter_shape (list): A list of
            `[height, width, input_channel, output_channel]`
        name (string): A layer name
    Returns:
        outputs: A tensor of size `[B * T, H, W, output_channel]`
    """
    with tf.variable_scope(name):
        W = tf.Variable(tf.truncated_normal(shape=filter_shape,
                                            stddev=parameter_init),
                        name='weight')
        b = tf.Variable(tf.zeros(shape=filter_shape[-1]), name='bias')
        conv_bottom = tf.nn.conv2d(bottom, W,
                                   strides=[1, 1, 1, 1],
                                   padding='SAME')
        outputs = tf.nn.bias_add(conv_bottom, b)

        if not relu:
            return outputs

        return tf.nn.relu(outputs)
