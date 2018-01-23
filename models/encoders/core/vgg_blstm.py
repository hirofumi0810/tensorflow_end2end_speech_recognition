#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""VGG + bidirectional LSTM encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from models.encoders.core.cnn_util import conv_layer, max_pool, batch_normalization
from models.encoders.core.blstm import basiclstmcell, lstmcell, lstmblockcell, lstmblockfusedcell, cudnnlstm


class VGGBLSTMEncoder(object):
    """VGG + bidirectional LSTM encoder.
    Args:
        input_size (int): the dimensions of input vectors．
            This is expected to be num_channels * 3 (static + Δ + ΔΔ)
        splice (int): frames to splice
        num_stack (int): the number of frames to stack
        num_units (int): the number of units in each layer
        num_proj (int): the number of nodes in the projection layer
        num_layers (int): the number of layers
        lstm_impl (string, optional): a base implementation of LSTM.
                - BasicLSTMCell: tf.contrib.rnn.BasicLSTMCell (no peephole)
                - LSTMCell: tf.contrib.rnn.LSTMCell
                - LSTMBlockCell: tf.contrib.rnn.LSTMBlockCell
                - LSTMBlockFusedCell: under implementation
                - CudnnLSTM: under implementation
            Choose the background implementation of tensorflow.
        use_peephole (bool): if True, use peephole
        parameter_init (float): the range of uniform distribution to
            initialize weight parameters (>= 0)
        clip_activation (float): the range of activation clipping (> 0)
        time_major (bool, optional): if True, time-major computation will be
            performed
        name (string, optional): the name of encoder
    """

    def __init__(self,
                 input_size,
                 splice,
                 num_stack,
                 num_units,
                 num_proj,
                 num_layers,
                 lstm_impl,
                 use_peephole,
                 parameter_init,
                 clip_activation,
                 time_major=False,
                 name='vgg_blstm_encoder'):

        assert num_proj != 0
        assert input_size % 3 == 0

        self.num_channels = input_size // 3
        self.splice = splice
        self.num_stack = num_stack
        self.num_units = num_units
        if lstm_impl != 'LSTMCell':
            self.num_proj = None
        else:
            self.num_proj = num_proj
        # TODO: fix this
        self.num_layers = num_layers
        self.lstm_impl = lstm_impl
        self.use_peephole = use_peephole
        self.parameter_init = parameter_init
        self.clip_activation = clip_activation
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
                if time_major is True, a tensor of size
                    `[T, B, num_units (num_proj)]`
                otherwise, `[B, T, num_units (num_proj)]`
            final_state: A final hidden state of the encoder
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
            inputs = conv_layer(inputs,
                                filter_size=[3, 3, 3, 64],
                                stride=[1, 1],
                                parameter_init=self.parameter_init,
                                activation='relu',
                                name='conv1')
            # inputs = batch_normalization(inputs, is_training=is_training)
            inputs = tf.nn.dropout(inputs, keep_prob)

            inputs = conv_layer(inputs,
                                filter_size=[3, 3, 64, 64],
                                stride=[1, 1],
                                parameter_init=self.parameter_init,
                                activation='relu',
                                name='conv2')
            # inputs = batch_normalization(inputs, is_training=is_training)
            inputs = max_pool(inputs,
                              pooling_size=[2, 2],
                              stride=[2, 2],
                              name='max_pool')
            inputs = tf.nn.dropout(inputs, keep_prob)

        with tf.variable_scope('VGG2'):
            inputs = conv_layer(inputs,
                                filter_size=[3, 3, 64, 128],
                                stride=[1, 1],
                                parameter_init=self.parameter_init,
                                activation='relu',
                                name='conv1')
            # inputs = batch_normalization(inputs, is_training=is_training)
            inputs = tf.nn.dropout(inputs, keep_prob)

            inputs = conv_layer(inputs,
                                filter_size=[3, 3, 128, 128],
                                stride=[1, 1],
                                parameter_init=self.parameter_init,
                                activation='relu',
                                name='conv2')
            # inputs = batch_normalization(inputs, is_training=is_training)
            inputs = max_pool(inputs,
                              pooling_size=[2, 2],
                              stride=[2, 2],
                              name='max_pool')
            inputs = tf.nn.dropout(inputs, keep_prob)

        # Reshape to 2D tensor `[B * T, new_h * new_w * C_out]`
        inputs = tf.reshape(
            inputs, shape=[batch_size * max_time, np.prod(inputs.shape.as_list()[-3:])])

        # Insert linear layer to recude CNN's output demention
        # from (new_h * new_w * C_out) to 256
        with tf.variable_scope('bridge') as scope:
            inputs = tf.contrib.layers.fully_connected(
                inputs=inputs,
                num_outputs=256,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.truncated_normal_initializer(
                    stddev=self.parameter_init),
                biases_initializer=tf.zeros_initializer(),
                scope=scope)
            inputs = tf.nn.dropout(inputs, keep_prob)

        # Reshape back to 3D tensor `[B, T, 256]`
        inputs = tf.reshape(inputs, shape=[batch_size, max_time, 256])

        initializer = tf.random_uniform_initializer(
            minval=-self.parameter_init, maxval=self.parameter_init)

        if self.lstm_impl == 'BasicLSTMCell':
            outputs, final_state = basiclstmcell(
                self.num_units, self.num_layers,
                inputs, inputs_seq_len, keep_prob, initializer,
                self.time_major)

        elif self.lstm_impl == 'LSTMCell':
            outputs, final_state = lstmcell(
                self.num_units, self.num_proj, self.num_layers,
                self.use_peephole, self.clip_activation,
                inputs, inputs_seq_len, keep_prob, initializer,
                self.time_major)

        elif self.lstm_impl == 'LSTMBlockCell':
            outputs, final_state = lstmblockcell(
                self.num_units, self.num_layers,
                self.use_peephole, self.clip_activation,
                inputs, inputs_seq_len, keep_prob, initializer,
                self.time_major)

        elif self.lstm_impl == 'LSTMBlockFusedCell':
            outputs, final_state = lstmblockfusedcell(
                self.num_units, self.num_layers,
                self.use_peephole, self.clip_activation,
                inputs, inputs_seq_len, keep_prob, initializer,
                self.time_major)

        elif self.lstm_impl == 'CudnnLSTM':
            outputs, final_state = cudnnlstm(
                self.num_units, self.num_layers, self.parameter_init,
                inputs, inputs_seq_len, keep_prob, initializer,
                self.time_major)
        else:
            raise IndexError(
                'lstm_impl is "BasicLSTMCell" or "LSTMCell" or ' +
                '"LSTMBlockCell" or "LSTMBlockFusedCell" or ' +
                '"CudnnLSTM".')

        return outputs, final_state
