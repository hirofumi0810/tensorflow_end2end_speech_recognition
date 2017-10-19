#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""VGG encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
# import numpy as np
import tensorflow as tf

from models.encoders.core.cnn_util import conv_layer, max_pool, batch_normalization

############################################################
# Architecture: (feature map, kernel, stride)
# VGG1: (96, 3*3, (1,1))  * 3 layers
# Batch normalization
# ReLU
# Max pool
# VGG2: (192, 3*3, (1,1)) * 4 layers
# Batch normalization
# ReLU
# Max pool
# VGG3: (384, 3*3, (1,1)) * 4 layers
# Batch normalization
# ReLU
# Max pool
# (fc: 1024 * 2 layers) ??
############################################################


class VGGEncoder(object):
    """VGG encoder.
       This implementation is based on
        https://arxiv.org/abs/1702.07793.
            Wang, Yisen, et al.
            "Residual convolutional CTC networks for automatic speech recognition."
            arXiv preprint arXiv:1702.07793 (2017).
    Args:
        input_size (int): the dimensions of input vectors.
            This is expected to be num_channels * 3 (static + Δ + ΔΔ)
        splice (int): frames to splice
        parameter_init (float, optional): the range of uniform distribution to
            initialize weight parameters (>= 0)
        time_major (bool, optional): if True, time-major computation will be
            performed
        name (string, optional): the name of encoder
    """

    def __init__(self,
                 input_size,
                 splice,
                 parameter_init,
                 time_major,
                 name='vgg_encoder'):

        assert input_size % 3 == 0

        self.input_size = input_size
        self.num_channels = input_size // 3
        self.splice = splice
        self.parameter_init = parameter_init
        self.time_major = time_major
        self.name = name

    def __call__(self, inputs, inputs_seq_len, keep_prob):
        """Construct model graph.
        Args:
            inputs (placeholder): A tensor of size
                `[B, T, input_size (num_channels * splice * 3)]`
            inputs_seq_len (placeholder): A tensor of size` [B]`
            keep_prob (placeholder, float): A probability to keep nodes
                in the hidden-hidden connection
        Returns:
            outputs: Encoder states.
                if time_major is True, a tensor of size `[T, B, output_dim]`
                otherwise, `[B, T, output_dim]`
            final_state: None
        """
        # inputs: 3D tensor `[B, T, input_size (num_channels * splice * 3)]`
        batch_size = tf.shape(inputs)[0]
        max_time = tf.shape(inputs)[1]

        # Reshape to 4D tensor `[B * T, num_channels, splice, 3]`
        inputs = tf.reshape(
            inputs,
            shape=[batch_size * max_time, self.num_channels, self.splice, 3])

        # NOTE: filter_size: `[H, W, C_in, C_out]`
        with tf.variable_scope('VGG1'):
            inputs = conv_layer(inputs,
                                filter_size=[3, 3, 3, 96],
                                stride=[1, 1],
                                parameter_init=self.parameter_init,
                                activation='relu',
                                name='conv1')
            inputs = batch_normalization(inputs, is_training=True)
            inputs = conv_layer(inputs,
                                filter_size=[3, 3, 96, 96],
                                stride=[1, 1],
                                parameter_init=self.parameter_init,
                                activation='relu',
                                name='conv2')
            inputs = batch_normalization(inputs, is_training=True)
            inputs = conv_layer(inputs,
                                filter_size=[3, 3, 96, 96],
                                stride=[1, 1],
                                parameter_init=self.parameter_init,
                                activation='relu',
                                name='conv3')
            inputs = batch_normalization(inputs, is_training=True)
            inputs = max_pool(inputs, name='max_pool')

        with tf.variable_scope('VGG2'):
            inputs = conv_layer(inputs,
                                filter_size=[3, 3, 96, 192],
                                stride=[1, 1],
                                parameter_init=self.parameter_init,
                                activation='relu',
                                name='conv1')
            inputs = batch_normalization(inputs, is_training=True)
            inputs = conv_layer(inputs,
                                filter_size=[3, 3, 192, 192],
                                stride=[1, 1],
                                parameter_init=self.parameter_init,
                                activation='relu',
                                name='conv2')
            inputs = batch_normalization(inputs, is_training=True)
            inputs = conv_layer(inputs,
                                filter_size=[3, 3, 192, 192],
                                stride=[1, 1],
                                parameter_init=self.parameter_init,
                                activation='relu',
                                name='conv3')
            inputs = batch_normalization(inputs, is_training=True)
            inputs = conv_layer(inputs,
                                filter_size=[3, 3, 192, 192],
                                stride=[1, 1],
                                parameter_init=self.parameter_init,
                                activation='relu',
                                name='conv4')
            inputs = batch_normalization(inputs, is_training=True)
            inputs = max_pool(inputs, name='max_pool')

        with tf.variable_scope('VGG3'):
            inputs = conv_layer(inputs,
                                filter_size=[3, 3, 192, 384],
                                parameter_init=self.parameter_init,
                                activation='relu',
                                name='conv1')
            inputs = batch_normalization(inputs, is_training=True)
            inputs = conv_layer(inputs,
                                filter_size=[3, 3, 384, 384],
                                parameter_init=self.parameter_init,
                                activation='relu',
                                name='conv2')
            inputs = batch_normalization(inputs, is_training=True)
            inputs = conv_layer(inputs,
                                filter_size=[3, 3, 384, 384],
                                parameter_init=self.parameter_init,
                                activation='relu',
                                name='conv3')
            inputs = batch_normalization(inputs, is_training=True)
            inputs = conv_layer(inputs,
                                filter_size=[3, 3, 384, 384],
                                parameter_init=self.parameter_init,
                                activation='relu',
                                name='conv4')
            inputs = batch_normalization(inputs, is_training=True)
            inputs = max_pool(inputs, name='max_pool')

        # Reshape to 2D tensor `[B * T, new_h * new_w * 384]`
        new_h = math.ceil(self.num_channels / (2**3))  # 5 or 6
        new_w = math.ceil(self.splice / (2**3))  # 2
        outputs = tf.reshape(
            inputs, shape=[batch_size * max_time, new_h * new_w * 384])

        with tf.variable_scope('fc') as scope:
            outputs = tf.contrib.layers.fully_connected(
                inputs=outputs,
                num_outputs=1024,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.truncated_normal_initializer(
                    stddev=self.parameter_init),
                biases_initializer=tf.zeros_initializer(),
                scope=scope)

        # Reshape back to 3D tensor `[B, T, 1024]`
        outputs = tf.reshape(
            outputs, shape=[batch_size, max_time, 1024])

        if self.time_major:
            # Convert to time-major: `[T, B, num_classes]'
            outputs = tf.transpose(outputs, [1, 0, 2])

        return outputs, None
