#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""VGG encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from models.encoders.core.cnn_util import conv_layer, max_pool, batch_normalization

############################################################
# Architecture: (feature map, kernel, stride)

# VGG1: (96, 3*3, (1,1)) * 3 layers
# Batch normalization
# ReLU
# Max pool
# dropout

# VGG2: (192, 3*3, (1,1)) * 4 layers
# Batch normalization
# ReLU
# Max pool
# dropout

# VGG3: (384, 3*3, (1,1)) * 4 layers
# Batch normalization
# ReLU
# Max pool
# dropout

# fc: 1024 * 2 layers
# (dropout, first layer only)

# softmax
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
        num_stack (int): the number of frames to stack
        parameter_init (float, optional): the range of uniform distribution to
            initialize weight parameters (>= 0)
        time_major (bool, optional): if True, time-major computation will be
            performed
        name (string, optional): the name of encoder
    """

    def __init__(self,
                 input_size,
                 splice,
                 num_stack,
                 parameter_init,
                 time_major,
                 name='vgg_wang_encoder'):

        assert input_size % 3 == 0

        self.num_channels = input_size // 3
        self.splice = splice
        self.num_stack = num_stack
        self.parameter_init = parameter_init
        self.time_major = time_major
        self.name = name

    def __call__(self, inputs, inputs_seq_len, keep_prob, is_training):
        """Construct model graph.
        Args:
            inputs (placeholder): A tensor of size
                `[B, T, input_size (num_channels * (splice * num_stack) * 3)]`
            inputs_seq_len (placeholder): A tensor of size` [B]`
            keep_prob (placeholder, float): A probability to keep nodes
                in the hidden-hidden connection
            is_training (bool):
        Returns:
            outputs: Encoder states.
                if time_major is True, a tensor of size `[T, B, output_dim]`
                otherwise, `[B, T, output_dim]`
            final_state: None
        """
        # inputs: 3D tensor `[B, T, input_dim]`
        batch_size = tf.shape(inputs)[0]
        max_time = tf.shape(inputs)[1]
        input_dim = inputs.shape.as_list()[-1]
        # NOTE: input_dim: num_channels * splice * num_stack * 3

        # For debug
        # print(input_dim)
        # print(self.num_channels)
        # print(self.splice)
        # print(self.num_stack)

        assert input_dim == self.num_channels * self.splice * self.num_stack * 3

        # Reshape to 4D tensor `[B * T, num_channels, splice * num_stack, 3]`
        inputs = tf.reshape(
            inputs,
            shape=[batch_size * max_time, self.num_channels, self.splice * self.num_stack, 3])

        # NOTE: filter_size: `[H, W, C_in, C_out]`
        with tf.variable_scope('VGG1'):
            for i_layer in range(1, 4, 1):
                input_channels = inputs.shape.as_list()[-1]
                inputs = conv_layer(inputs,
                                    filter_size=[3, 3, input_channels, 96],
                                    stride=[1, 1],
                                    parameter_init=self.parameter_init,
                                    activation='relu',
                                    name='conv1')
                inputs = batch_normalization(inputs, is_training=is_training)
                if i_layer == 3:
                    inputs = max_pool(inputs,
                                      pooling_size=[2, 2],
                                      stride=[2, 2],
                                      name='max_pool')
                inputs = tf.nn.dropout(inputs, keep_prob)

        with tf.variable_scope('VGG2'):
            for i_layer in range(1, 5, 1):
                input_channels = inputs.shape.as_list()[-1]
                inputs = conv_layer(inputs,
                                    filter_size=[3, 3, input_channels, 192],
                                    stride=[1, 1],
                                    parameter_init=self.parameter_init,
                                    activation='relu',
                                    name='conv%d' % i_layer)
                inputs = batch_normalization(inputs, is_training=is_training)
                if i_layer == 4:
                    inputs = max_pool(inputs,
                                      pooling_size=[2, 2],
                                      stride=[2, 2],
                                      name='max_pool')
                inputs = tf.nn.dropout(inputs, keep_prob)

        with tf.variable_scope('VGG3'):
            for i_layer in range(1, 5, 1):
                input_channels = inputs.shape.as_list()[-1]
                inputs = conv_layer(inputs,
                                    filter_size=[3, 3, input_channels, 384],
                                    parameter_init=self.parameter_init,
                                    activation='relu',
                                    name='conv%d' % i_layer)
                inputs = batch_normalization(inputs, is_training=is_training)
                if i_layer == 4:
                    inputs = max_pool(inputs,
                                      pooling_size=[2, 2],
                                      stride=[2, 2],
                                      name='max_pool')
                inputs = tf.nn.dropout(inputs, keep_prob)

        # Reshape to 2D tensor `[B * T, new_h * new_w * C_out]`
        outputs = tf.reshape(
            inputs, shape=[batch_size * max_time, np.prod(inputs.shape.as_list()[-3:])])

        for i_layer in range(1, 3, 1):
            with tf.variable_scope('fc%d' % i_layer) as scope:
                outputs = tf.contrib.layers.fully_connected(
                    inputs=outputs,
                    num_outputs=1024,
                    activation_fn=tf.nn.relu,
                    weights_initializer=tf.truncated_normal_initializer(
                        stddev=self.parameter_init),
                    biases_initializer=tf.zeros_initializer(),
                    scope=scope)
                if i_layer == 1:
                    outputs = tf.nn.dropout(outputs, keep_prob)

        # Reshape back to 3D tensor `[B, T, 1024]`
        output_dim = outputs.shape.as_list()[-1]
        outputs = tf.reshape(
            outputs, shape=[batch_size, max_time, output_dim])

        if self.time_major:
            # Convert to time-major: `[T, B, num_classes]'
            outputs = tf.transpose(outputs, [1, 0, 2])

        return outputs, None
