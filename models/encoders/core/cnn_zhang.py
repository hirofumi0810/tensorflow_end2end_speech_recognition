#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""CNN encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from models.encoders.core.cnn_util import conv_layer, max_pool, batch_normalization

############################################################
# Architecture: (feature map, kernel, stride)

# CNN1: (128, 3*5, (1,1)) * 1 layer
# ReLU
# (Batch normalization)
# max pooling
# dropout

# CNN2-5: (128, 3*5, (1,1)) * 4 layer
# ReLU
# (Batch normalization)
# dropout

# CNN6-10: (256, 3*5, (1,1)) * 5 layer
# ReLU
# (Batch normalization)
# dropout

# fc: 1024 * 3 layer
# (dropout, except for the last layer)

# softmax
############################################################


class CNNEncoder(object):
    """CNN encoder.
       This implementation is based on
           https://arxiv.org/abs/1701.02720.
               Zhang, Ying, et al.
               "Towards end-to-end speech recognition with deep convolutional
                neural networks."
               arXiv preprint arXiv:1701.02720 (2017).
    Args:
        input_size (int): the dimensions of input vectors
        splice (int): frames to splice. Default is 1 frame.
        num_stack (int): the number of frames to stack
        parameter_init (float, optional): Range of uniform distribution to
            initialize weight parameters
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
                 name='cnn_zhang_encoder'):

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
                `[B, T, input_size (num_channels * splice * num_stack * 3)]`
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

        # Choose the activation function
        activation = 'relu'
        # activation = 'prelu'
        # activation = 'maxout'
        # TODO: add prelu and maxout layers

        # NOTE: filter_size: `[H, W, C_in, C_out]`
        # 1-4th layers
        for i_layer in range(1, 5, 1):
            with tf.variable_scope('CNN%d' % i_layer):
                input_channels = inputs.shape.as_list()[-1]
                inputs = conv_layer(inputs,
                                    filter_size=[3, 5, input_channels, 128],
                                    stride=[1, 1],
                                    parameter_init=self.parameter_init,
                                    activation=activation,
                                    name='conv')
                # inputs = batch_normalization(inputs, is_training=is_training)
                if i_layer == 1:
                    inputs = max_pool(inputs,
                                      pooling_size=[3, 1],
                                      stride=[3, 1],
                                      name='pool')
                inputs = tf.nn.dropout(inputs, keep_prob)

        # 5-10th layers
        for i_layer in range(5, 11, 1):
            with tf.variable_scope('CNN%d' % i_layer):
                input_channels = inputs.shape.as_list()[-1]
                inputs = conv_layer(inputs,
                                    filter_size=[3, 5, input_channels, 256],
                                    stride=[1, 1],
                                    parameter_init=self.parameter_init,
                                    activation=activation,
                                    name='conv')
                # inputs = batch_normalization(inputs, is_training=is_training)
                # NOTE: No poling
                inputs = tf.nn.dropout(inputs, keep_prob)

        # Reshape to 2D tensor `[B * T, new_h * new_w * C_out]`
        outputs = tf.reshape(
            inputs, shape=[batch_size * max_time, np.prod(inputs.shape.as_list()[-3:])])

        # 11-14th layers
        for i_layer in range(1, 4, 1):
            with tf.variable_scope('fc%d' % i_layer) as scope:
                outputs = tf.contrib.layers.fully_connected(
                    inputs=outputs,
                    num_outputs=1024,
                    activation_fn=tf.nn.relu,
                    weights_initializer=tf.truncated_normal_initializer(
                        stddev=self.parameter_init),
                    biases_initializer=tf.zeros_initializer(),
                    scope=scope)
                if i_layer != 3:
                    outputs = tf.nn.dropout(outputs, keep_prob)

        # Reshape back to 3D tensor `[B, T, 1024]`
        logits = tf.reshape(
            outputs, shape=[batch_size, max_time, 1024])

        if self.time_major:
            # Convert to time-major: `[T, B, 1024]'
            logits = tf.transpose(logits, [1, 0, 2])

        return logits, None
