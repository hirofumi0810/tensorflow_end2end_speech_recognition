#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""VGG encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
# import numpy as np
import tensorflow as tf

from models.encoders.core.cnn_util import conv_layer, max_pool

############################################################
# Architecture
#      (feature map, kernel)
# VGG1 (96, 3*3)  * 3 layers
# VGG2 (192, 3*3) * 4 layers
# VGG3 (384, 3*3) * 4 layers
# fc      1024    * 3
############################################################


class VGG_Encoder(object):
    """VGG encoder.
       This implementation is based on
        https://arxiv.org/abs/1702.07793.
            Wang, Yisen, et al.
            "Residual convolutional CTC networks for automatic speech recognition."
            arXiv preprint arXiv:1702.07793 (2017).
    Args:
        input_size (int): the dimensions of input vectors
        splice (int): frames to splice
        # num_units (int): the number of units in each layer
        # num_layers (int): the number of layers
        num_classes (int): the number of classes of target labels
            (except for a blank label). if 0, return hidden states before
            passing through the softmax layer
        parameter_init (float, optional): the range of uniform distribution to
            initialize weight parameters (>= 0)
        clip_activation (float, optional): the range of activation clipping (> 0)
        bottleneck_dim (int, optional): the dimensions of the bottleneck layer
        name (string, optional): the name of encoder
    """

    def __init__(self,
                 input_size,
                 splice,
                 num_classes,
                 parameter_init=0.1,
                 clip_activation=5.0,
                 bottleneck_dim=None,
                 name='vgg_encoder'):

        self.input_size = input_size
        self.splice = splice
        self.num_classes = num_classes
        self.parameter_init = parameter_init
        self.clip_activation = clip_activation
        self.bottleneck_dim = int(bottleneck_dim) if bottleneck_dim not in [
            None, 0] else None
        self.name = name

    def __call__(self, inputs, inputs_seq_len,
                 keep_prob_input, keep_prob_hidden, keep_prob_output):
        """Construct model graph.
        Args:
            inputs (placeholder): A tensor of size`[B, T, input_size]`
            inputs_seq_len (placeholder): A tensor of size` [B]`
            keep_prob_input (placeholder, float): A probability to keep nodes
                in the input-hidden connection
            keep_prob_hidden (placeholder, float): A probability to keep nodes
                in the hidden-hidden connection
            keep_prob_output (placeholder, float): A probability to keep nodes
                in the hidden-output connection
        Returns:
            logits: A tensor of size `[T, B, num_classes]`
            final_state: A final hidden state of the encoder
        """
        # inputs: 3D tensor `[batch_size, max_time, input_size * splice]`
        batch_size = tf.shape(inputs)[0]
        max_time = tf.shape(inputs)[1]

        # Reshape to 4D tensor
        # `[batch_size * max_time, input_size / 3, splice, 3(+Δ,ΔΔ)]`
        inputs = tf.reshape(
            inputs,
            shape=[batch_size * max_time, int(self.input_size / 3), self.splice, 3])

        with tf.variable_scope('VGG1'):
            inputs = conv_layer(inputs,
                                filter_shape=[3, 3, 3, 96],
                                parameter_init=self.parameter_init,
                                name='conv1')
            inputs = conv_layer(inputs,
                                filter_shape=[3, 3, 96, 96],
                                parameter_init=self.parameter_init,
                                name='conv2')
            inputs = conv_layer(inputs,
                                filter_shape=[3, 3, 96, 96],
                                parameter_init=self.parameter_init,
                                name='conv3')
            inputs = max_pool(inputs, name='max_pool')
            # TODO(hirofumi): try batch normalization

        with tf.variable_scope('VGG2'):
            inputs = conv_layer(inputs,
                                filter_shape=[3, 3, 96, 192],
                                parameter_init=self.parameter_init,
                                name='conv1')
            inputs = conv_layer(inputs,
                                filter_shape=[3, 3, 192, 192],
                                parameter_init=self.parameter_init,
                                name='conv2')
            inputs = conv_layer(inputs,
                                filter_shape=[3, 3, 192, 192],
                                parameter_init=self.parameter_init,
                                name='conv3')
            inputs = conv_layer(inputs,
                                filter_shape=[3, 3, 192, 192],
                                parameter_init=self.parameter_init,
                                name='conv4')
            inputs = max_pool(inputs, name='max_pool')
            # TODO(hirofumi): try batch normalization

        with tf.variable_scope('VGG3'):
            inputs = conv_layer(inputs,
                                filter_shape=[3, 3, 192, 384],
                                parameter_init=self.parameter_init,
                                name='conv1')
            inputs = conv_layer(inputs,
                                filter_shape=[3, 3, 384, 384],
                                parameter_init=self.parameter_init,
                                name='conv2')
            inputs = conv_layer(inputs,
                                filter_shape=[3, 3, 384, 384],
                                parameter_init=self.parameter_init,
                                name='conv3')
            inputs = conv_layer(inputs,
                                filter_shape=[3, 3, 384, 384],
                                parameter_init=self.parameter_init,
                                name='conv4')
            inputs = max_pool(inputs, name='max_pool')
            # TODO(hirofumi): try batch normalization

        # Reshape to 2D tensor `[batch_size * max_time, new_h * new_w * 384]`
        new_h = math.ceil(self.input_size / (3 * 2**3))  # expected to be 5 or 6
        new_w = math.ceil(self.splice / (2**3))  # expected to be 2
        inputs = tf.reshape(
            inputs, shape=[batch_size * max_time, new_h * new_w * 384])

        with tf.variable_scope('fc1') as scope:
            inputs = tf.contrib.layers.fully_connected(
                inputs=inputs,
                num_outputs=1024,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.truncated_normal_initializer(
                    stddev=self.parameter_init),
                biases_initializer=tf.zeros_initializer(),
                scope=scope)

        with tf.variable_scope('fc2') as scope:
            inputs = tf.contrib.layers.fully_connected(
                inputs=inputs,
                num_outputs=1024,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.truncated_normal_initializer(
                    stddev=self.parameter_init),
                biases_initializer=tf.zeros_initializer(),
                scope=scope)

        with tf.variable_scope('fc3') as scope:
            logits_2d = tf.contrib.layers.fully_connected(
                inputs=inputs,
                num_outputs=self.num_classes,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.truncated_normal_initializer(
                    stddev=self.parameter_init),
                biases_initializer=tf.zeros_initializer(),
                scope=scope)

        # if self.bottleneck_dim is not None and self.bottleneck_dim != 0:
        #     with tf.variable_scope('bottleneck') as scope:
        #         outputs = tf.contrib.layers.fully_connected(
        #             outputs, self.bottleneck_dim,
        #             activation_fn=tf.nn.relu,
        #             weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
        #             biases_initializer=tf.zeros_initializer(),
        #             scope=scope)
        #
        #         # Dropout for the hidden-output connections
        #         outputs = tf.nn.dropout(
        #             outputs, keep_prob_output, name='dropout_output_bottle')

        # Reshape back to 3D tensor `[batch_size, max_time, num_classes]`
        logits = tf.reshape(
            logits_2d, shape=[batch_size, max_time, self.num_classes])

        # Convert to time-major: `[max_time, batch_size, num_classes]'
        logits = tf.transpose(logits, (1, 0, 2))

        return logits, None
