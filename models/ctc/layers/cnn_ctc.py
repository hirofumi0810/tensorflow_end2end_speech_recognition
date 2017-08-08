#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""CNN-CTC model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from models.ctc.ctc_base import ctcBase


class CNN_CTC(ctcBase):
    """CNN-CTC model.
       This implementaion is based on
           https://arxiv.org/abs/1701.02720.
               Zhang, Ying, et al.
               "Towards end-to-end speech recognition with deep convolutional
                neural networks."
               arXiv preprint arXiv:1701.02720 (2017).
    Args:
        input_size: int, the dimensions of input vectors
        num_unit: int, the number of units in each layer
        num_layer: int, the number of layers
        num_classes: int, the number of classes of target labels
            (except for a blank label)
        splice: int, frames to splice. Default is 1 frame.
        parameter_init: A float value. Range of uniform distribution to
            initialize weight parameters
        clip_grad: A float value. Range of gradient clipping (> 0)
        clip_activation: A float value. Range of activation clipping (> 0)
        dropout_ratio_input: A float value. Dropout ratio in the input-hidden
            layer
        dropout_ratio_hidden: A float value. Dropout ratio in the hidden-hidden
            layers
        dropout_ratio_output: A float value. Dropout ratio in the hidden-output
            layer
        num_proj: not used
        weight_decay: A float value. Regularization parameter for weight decay
        bottleneck_dim: not used
    """

    def __init__(self,
                 input_size,
                 num_classes,
                 num_unit=0,  # not used
                 num_layer=0,  # not used
                 splice=1,
                 parameter_init=0.1,
                 clip_grad=None,
                 clip_activation=None,
                 dropout_ratio_input=1.0,
                 dropout_ratio_hidden=1.0,
                 dropout_ratio_output=1.0,
                 num_proj=None,  # not used
                 weight_decay=0.0,
                 bottleneck_dim=None,  # not used
                 name='cnn_ctc'):

        ctcBase.__init__(self, input_size, num_unit, num_layer, num_classes,
                         splice, parameter_init, clip_grad, clip_activation,
                         dropout_ratio_input, dropout_ratio_hidden,
                         dropout_ratio_output, weight_decay, name)

    def _build(self, inputs, inputs_seq_len, keep_prob_hidden,
               keep_prob_input=None, keep_prob_output=None):
        """Construct model graph.
        Args:
            inputs: A tensor of size `[B, T, input_size]`
            inputs_seq_len:  A tensor of size `[B]`
            keep_prob_input: A float value. A probability to keep nodes in
                the input-hidden layer
            keep_prob_hidden: A float value. A probability to keep nodes in
                the hidden-hidden layers
            keep_prob_output: A float value. A probability to keep nodes in
                the hidden-output layer
        Returns:
            logits: A tensor of size `[T, B, num_classes]`
        """
        # NOTE: input dropout is not performed

        # inputs: 3D `[batch_size, max_time, input_size * splice]`
        batch_size = tf.shape(inputs)[0]
        max_time = tf.shape(inputs)[1]

        # Reshape to 4D `[batch_size, max_time, input_size, splice]`
        inputs = tf.reshape(
            inputs, shape=[batch_size, max_time, self.input_size, self.splice])

        # Reshape to 5D `[batch_size, max_time, input_size / 3, splice, 3 (+Δ,
        # ΔΔ)]`
        inputs = tf.reshape(
            inputs, shape=[batch_size, max_time, int(self.input_size / 3), 3, self.splice])
        inputs = tf.transpose(inputs, (0, 1, 2, 4, 3))

        # Reshape to 4D `[batch_size * max_time, input_size / 3, splice, 3]`
        inputs = tf.reshape(
            inputs, shape=[batch_size * max_time, int(self.input_size / 3), self.splice, 3])

        # Choose the activation function
        activation = 'relu'
        # activation = 'prelu'
        # activation = 'maxout'

        # 1-4 layers
        with tf.name_scope('conv128'):
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
        with tf.name_scope('conv256'):
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

        # Reshape to 5D `[batch_size, max_time, 14, splice, 256]`
        outputs = tf.reshape(
            outputs, shape=[batch_size, max_time, math.ceil(self.input_size / 3 / 3), self.splice, 256])

        # Reshape to 3D `[batch_size, max_time, 14 * splice * 256]`
        outputs = tf.reshape(
            outputs, shape=[batch_size, max_time, math.ceil(self.input_size / 3 / 3) * self.splice * 256])

        # Reshape to 2D `[batch_size * max_time, 14 * splice * 256]`
        outputs = tf.reshape(
            outputs, shape=[batch_size * max_time, math.ceil(self.input_size / 3 / 3) * self.splice * 256])

        # 11~13th fc
        with tf.name_scope('fc'):
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

        # Reshape back to 3D `[batch_size, max_time, num_classes]`
        logits = tf.reshape(
            outputs, shape=[batch_size, max_time, self.num_classes])

        # Convert to time-major: `[max_time, batch_size, num_classes]'
        logits = tf.transpose(logits, (1, 0, 2))

        return logits

    def _max_pool(self, bottom, name):
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
