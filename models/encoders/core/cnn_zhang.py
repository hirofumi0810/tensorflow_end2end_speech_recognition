#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""CNN encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf


class CNN_Encoder(object):
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
        num_classes (int): the number of classes of target labels
            (except for a blank label)
        parameter_init (float, optional): Range of uniform distribution to
            initialize weight parameters
        name (string, optional): the name of encoder
    """

    def __init__(self,
                 input_size,
                 splice,
                 num_classes,
                 parameter_init=0.1,
                 name='cnn_encoder'):

        self.input_size = input_size
        self.splice = splice
        self.num_classes = num_classes
        self.parameter_init = parameter_init
        self.name = name

    def __call__(self, inputs, inputs_seq_len,
                 keep_prob_hidden, keep_prob_input, keep_prob_output):
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
        # TODO: input dropout is not performed

        # inputs: 3D `[batch_size, max_time, input_size * splice]`
        batch_size = tf.shape(inputs)[0]
        max_time = tf.shape(inputs)[1]

        # Reshape to 4D tensor `[batch_size, max_time, input_size, splice]`
        # inputs = tf.reshape(
        #     inputs,
        #     shape=[batch_size,
        #                    max_time,
        #                    self.input_size,
        #                    self.splice])

        # Reshape to 5D tensor
        # `[batch_size, max_time, input_size / 3, 3 (+Δ,ΔΔ), splice]`
        # inputs = tf.reshape(
        #     inputs, shape=[batch_size,
        #                    max_time,
        #                    int(self.input_size / 3),
        #                    3,
        #                    self.splice])

        # Reshape to 4D tensor
        # `[batch_size * max_time, input_size / 3, splice, 3]`
        # inputs = tf.transpose(inputs, (0, 1, 2, 4, 3))
        inputs = tf.reshape(
            inputs,
            shape=[batch_size * max_time, int(self.input_size / 3), self.splice, 3])

        # Choose the activation function
        activation = 'relu'
        # activation = 'prelu'
        # activation = 'maxout'
        # TODO: add prelu and maxout layers

        # 1-4 layers
        with tf.variable_scope('conv128'):
            for i_layer in range(1, 5, 1):
                if i_layer == 1:
                    outputs = self._conv_layer(inputs,
                                               filter_shape=[3, 5, 3, 128],
                                               name='conv%d' % i_layer)
                    outputs = self._activation(outputs, layer=activation)
                    outputs = self._max_pool(outputs, name='pool')
                else:
                    # no poling
                    outputs = self._conv_layer(outputs,
                                               filter_shape=[3, 5, 128, 128],
                                               name='conv%d' % i_layer)
                    outputs = self._activation(outputs, layer=activation)

                # Dropout
                outputs = tf.nn.dropout(outputs, keep_prob_hidden)
                # TODO: try Weight decay
                # TODO: try batch normalization

        # 5-10 layers
        with tf.variable_scope('conv256'):
            for i_layer in range(5, 11, 1):
                if i_layer == 5:
                    outputs = self._conv_layer(outputs,
                                               filter_shape=[3, 5, 128, 256],
                                               name='conv%d' % i_layer)
                else:
                    outputs = self._conv_layer(outputs,
                                               filter_shape=[3, 5, 256, 256],
                                               name='conv%d' % i_layer)

                outputs = self._activation(outputs, layer=activation)

                # Dropout
                outputs = tf.nn.dropout(outputs, keep_prob_hidden)
                # TODO: try Weight decay
                # TODO: try batch normalization

        # Reshape to 5D tensor `[batch_size, max_time, new_h, new_w, 256]`
        new_h = math.ceil(self.input_size / 3 / 3)  # expected to be 14
        new_w = self.splice  # expected to be 11
        # outputs = tf.reshape(
        #     outputs, shape=[batch_size, max_time, new_h, new_w, 256])

        # Reshape to 3D tensor `[batch_size, max_time, new_h * new_w * 256]`
        # outputs = tf.reshape(
        #     outputs, shape=[batch_size, max_time, new_h * new_w * 256])

        # Reshape to 2D tensor `[batch_size * max_time, new_h * new_w * 256]`
        outputs = tf.reshape(
            outputs, shape=[batch_size * max_time, new_h * new_w * 256])

        # 11-13th fc
        with tf.variable_scope('fc'):
            for i_layer in range(11, 14, 1):
                num_outputs = 1024 if i_layer != 13 else self.num_classes
                outputs = tf.contrib.layers.fully_connected(
                    inputs=outputs,
                    num_outputs=num_outputs,
                    activation_fn=tf.nn.relu,
                    scope='fc%d' % i_layer)

                if i_layer != 13:
                    # Dropout
                    outputs = tf.nn.dropout(outputs, keep_prob_hidden)
                # TODO: try Weight decay
                # TODO: try batch normalization

        # Reshape back to 3D tensor `[batch_size, max_time, num_classes]`
        logits = tf.reshape(
            outputs, shape=[batch_size, max_time, self.num_classes])

        # Convert to time-major: `[max_time, batch_size, num_classes]'
        logits = tf.transpose(logits, (1, 0, 2))

        return logits, None

    def _max_pool(self, bottom, name='max_pool'):
        """A max pooling layer.
        Args:
            bottom: A tensor of size `[B * T, H, W, C]`
            name: A layer name
        Returns:
            A tensor of size `[B * T, H / 3, W, C]`
        """
        return tf.nn.max_pool(
            bottom,
            ksize=[1, 3, 1, 1],
            strides=[1, 3, 1, 1],
            padding='SAME', name=name)

    def _conv_layer(self, bottom, filter_shape, name):
        """A convolutional layer
        Args:
            bottom: A tensor of size `[B * T, H, W, C]`
            filter_shape: A list of
                `[height, width, input_channel, output_channel]`
            name: A layer name
        Returns:
            outputs: A tensor of size `[B * T, H, W, output_channel]`
        """
        with tf.variable_scope(name):
            W = tf.Variable(tf.truncated_normal(shape=filter_shape,
                                                stddev=self.parameter_init),
                            name='weight')
            b = tf.Variable(tf.zeros(shape=filter_shape[-1]),
                            name='bias')
            conv_bottom = tf.nn.conv2d(bottom, W,
                                       strides=[1, 1, 1, 1],
                                       padding='SAME')
            return tf.nn.bias_add(conv_bottom, b)
            # NOTE: not performe activation

    def _activation(self, bottom, layer):
        """An activation layer.
        Args:
            bottom: A tensor of size `[B * T, H, W, C]`
            layer: relu or prelu or maxout
        Returns:
            outputs: A tensor of size `[B * T, H, W, C]`
        """
        if layer not in ['relu', 'prelu', 'maxout']:
            raise ValueError("layer is 'relu' or 'prelu' or 'maxout'.")

        if layer == 'relu':
            return tf.nn.relu(bottom)
        elif layer == 'prelu':
            raise NotImplementedError
        elif layer == 'maxout':
            raise NotImplementedError
