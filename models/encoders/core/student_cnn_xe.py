#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Student CNN encoder for XE training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from models.encoders.core.cnn_util import conv_layer, max_pool, batch_normalization

############################################################
# Architecture: (feature map, kernel(f*t), stride(f,t))

# CNN1: (128, 9*9, (1,1)) * 1 layers
# Batch normalization
# ReLU
# Max pool (3,1)

# CNN2: (256, 3*4, (1,1)) * 1 layers
# Batch normalization
# ReLU
# Max pool (1,1)

# fc: 2048 (ReLU) * 4 layers
############################################################


class StudentCNNXEEncoder(object):
    """Student CNN encoder for XE training.
    Args:
        input_size (int): the dimensions of input vectors.
            This is expected to be num_channels * 3 (static + Δ + ΔΔ)
        splice (int): frames to splice
        num_stack (int): the number of frames to stack
        parameter_init (float, optional): the range of uniform distribution to
            initialize weight parameters (>= 0)
        name (string, optional): the name of encoder
    """

    def __init__(self,
                 input_size,
                 splice,
                 num_stack,
                 parameter_init,
                 name='cnn_student_xe_encoder'):

        assert input_size % 3 == 0

        self.num_channels = (input_size // 3) // num_stack // splice
        self.splice = splice
        self.num_stack = num_stack
        self.parameter_init = parameter_init
        self.name = name

    def __call__(self, inputs, keep_prob, is_training):
        """Construct model graph.
        Args:
            inputs (placeholder): A tensor of size
                `[B, input_size (num_channels * splice * num_stack * 3)]`
            keep_prob (placeholder, float): A probability to keep nodes
                in the hidden-hidden connection
            is_training (bool):
        Returns:
            outputs: Encoder states.
                if time_major is True, a tensor of size `[T, B, output_dim]`
                otherwise, `[B, output_dim]`
        """
        # inputs: 2D tensor `[B, input_dim]`
        batch_size = tf.shape(inputs)[0]
        input_dim = inputs.shape.as_list()[-1]
        # NOTE: input_dim: num_channels * splice * num_stack * 3

        # for debug
        # print(input_dim)  # 1200
        # print(self.num_channels)  # 40
        # print(self.splice)  # 5
        # print(self.num_stack)  # 2

        assert input_dim == self.num_channels * self.splice * self.num_stack * 3

        # Reshape to 4D tensor `[B, num_channels, splice * num_stack, 3]`
        inputs = tf.reshape(
            inputs,
            shape=[batch_size, self.num_channels, self.splice * self.num_stack, 3])

        # NOTE: filter_size: `[H, W, C_in, C_out]`
        with tf.variable_scope('CNN1'):
            inputs = conv_layer(inputs,
                                filter_size=[9, 9, 3, 128],
                                stride=[1, 1],
                                parameter_init=self.parameter_init,
                                activation='relu')
            inputs = batch_normalization(inputs, is_training=is_training)
            inputs = max_pool(inputs,
                              pooling_size=[3, 1],
                              stride=[3, 1],
                              name='max_pool')

        with tf.variable_scope('CNN2'):
            inputs = conv_layer(inputs,
                                filter_size=[3, 4, 128, 256],
                                stride=[1, 1],
                                parameter_init=self.parameter_init,
                                activation='relu')
            inputs = batch_normalization(inputs, is_training=is_training)
            inputs = max_pool(inputs,
                              pooling_size=[1, 1],
                              stride=[1, 1],
                              name='max_pool')

        # Reshape to 2D tensor `[B, new_h * new_w * C_out]`
        outputs = tf.reshape(
            inputs, shape=[batch_size, np.prod(inputs.shape.as_list()[-3:])])

        for i in range(1, 5, 1):
            with tf.variable_scope('fc%d' % (i)) as scope:
                outputs = tf.contrib.layers.fully_connected(
                    inputs=outputs,
                    num_outputs=2048,
                    activation_fn=tf.nn.relu,
                    weights_initializer=tf.truncated_normal_initializer(
                        stddev=self.parameter_init),
                    biases_initializer=tf.zeros_initializer(),
                    scope=scope)

        return outputs
