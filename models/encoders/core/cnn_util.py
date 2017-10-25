#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilities for CNN-like layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def max_pool(bottom, pooling_size, stride=[2, 2], name='max_pool'):
    """A max pooling layer.
    Args:
        bottom: A tensor of size `[B (* T), H, W, C_in]`
        pooling_size (list): A list of `[pool_H, pool_w]`
        stride (list, optional): A list of `[stride_H, stride_W]`
        name (string, optional): A layer name
    Returns:
        outputs: A tensor of size `[B * T, H, W, C_out]`
    """
    return tf.nn.max_pool(
        bottom,
        ksize=[1, pooling_size[0], pooling_size[1], 1],
        strides=[1, stride[0], stride[1], 1],
        padding='SAME',
        name=name)


def avg_pool(bottom, pooling_size, stride=[2, 2], name='avg_pool'):
    """An average pooling layer.
    Args:
        bottom: A tensor of size `[B (* T), H, W, C_in]`
        pooling_size (list): A list of `[pool_H, pool_w]`
        stride (list, optional): A list of `[stride_H, stride_W]`
        name (string, optional): A layer name
    Returns:
        outputs: A tensor of size `[B * T, H, W, C_out]`
    """
    return tf.nn.avg_pool(
        bottom,
        ksize=[1, pooling_size[0], pooling_size[1], 1],
        strides=[1, stride[0], stride[1], 1],
        padding='SAME',
        name=name)


def conv_layer(bottom, filter_size, stride=[1, 1], parameter_init=0.1,
               activation=None, name='conv'):
    """A convolutional layer
    Args:
        bottom: A tensor of size `[B (* T), H, W, C_in]`
        filter_size (list): A list of `[H, W, C_in, C_out]`
        stride (list, optional): A list of `[stride_H, stride_W]`
        parameter_init (float, optional):
        activation (string, optional): relu
        name (string, optional): A layer name
    Returns:
        outputs: A tensor of size `[B * T, H, W, C_out]`
    """
    assert len(filter_size) == 4
    assert len(stride) == 2

    with tf.variable_scope(name):
        W = tf.Variable(tf.truncated_normal(shape=filter_size,
                                            stddev=parameter_init),
                        name='weight')
        b = tf.Variable(tf.zeros(shape=filter_size[-1]), name='bias')
        conv_bottom = tf.nn.conv2d(bottom, W,
                                   strides=[1, stride[0], stride[1], 1],
                                   padding='SAME')
        outputs = tf.nn.bias_add(conv_bottom, b)

        if activation is None:
            return outputs
        elif activation == 'relu':
            return tf.nn.relu(outputs)
        elif activation == 'prelu':
            raise NotImplementedError
        elif activation == 'maxout':
            raise NotImplementedError
        else:
            raise NotImplementedError


def batch_normalization(tensor, is_training=True, epsilon=0.001, momentum=0.9,
                        fused_batch_norm=False, name=None):
    """Performs batch normalization on given 4-D tensor.
    Args:
        tensor:
        epsilon:
        momentum:
        fused_batch_norm:
        name:
    Returns:

    The features are assumed to be in NHWC format. Noe that you need to
    run UPDATE_OPS in order for this function to perform correctly, e.g.:

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
      train_op = optimizer.minimize(loss)

    Based on: https://arxiv.org/abs/1502.03167
    """
    with tf.variable_scope(name, default_name="batch_norm"):
        channels = tensor.shape.as_list()[-1]
        axes = list(range(tensor.shape.ndims - 1))  # [0,1,2]

        beta = tf.get_variable(
            'beta', channels, initializer=tf.zeros_initializer())
        gamma = tf.get_variable(
            'gamma', channels, initializer=tf.ones_initializer())
        # NOTE: these are trainable parameters

        avg_mean = tf.get_variable(
            "avg_mean", channels, initializer=tf.zeros_initializer(),
            trainable=False)
        avg_variance = tf.get_variable(
            "avg_variance", channels, initializer=tf.ones_initializer(),
            trainable=False)

        if is_training:
            if fused_batch_norm:
                mean, variance = None, None
            else:
                mean, variance = tf.nn.moments(tensor, axes=axes)
        else:
            mean, variance = avg_mean, avg_variance

        if fused_batch_norm:
            tensor, mean, variance = tf.nn.fused_batch_norm(
                tensor, scale=gamma, offset=beta, mean=mean, variance=variance,
                epsilon=epsilon, is_training=is_training)
        else:
            tensor = tf.nn.batch_normalization(
                tensor, mean, variance, beta, gamma, epsilon)

        if is_training:
            update_mean = tf.assign(
                avg_mean, avg_mean * momentum + mean * (1.0 - momentum))
            update_variance = tf.assign(
                avg_variance, avg_variance * momentum + variance * (1.0 - momentum))

            # Ops before gradient update
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mean)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_variance)

    return tensor
