#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""VGG + bidirectional LSTM encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from models.encoders.core.cnn_util import conv_layer, max_pool
from models.encoders.core.blstm import basiclstmcell, lstmcell, lstmblockcell, lstmblockfusedcell, cudnnlstm


class VGGBLSTMEncoder(object):
    """VGG + bidirectional LSTM encoder.
    Args:
        input_size (int): the dimensions of input vectors
        splice (int): frames to splice
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
                 num_units,
                 num_proj,
                 num_layers,
                 lstm_impl,
                 use_peephole,
                 parameter_init,
                 clip_activation,
                 time_major=False,
                 name='vgg_blstm_encoder'):
        if num_proj == 0:
            raise ValueError

        self.input_size = input_size
        self.splice = splice
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

    def __call__(self, inputs, inputs_seq_len, keep_prob):
        """Construct model graph.
        Args:
            inputs (placeholder): A tensor of size`[B, T, input_size]`
            inputs_seq_len (placeholder): A tensor of size` [B]`
            keep_prob (placeholder, float): A probability to keep nodes
                in the hidden-hidden connection
        Returns:
            outputs: Encoder states, a tensor of size
                `[T, B, num_units (num_proj)]`
            final_state: A final hidden state of the encoder
        """
        # inputs: `[B, T, input_size * splice]`
        batch_size = tf.shape(inputs)[0]
        max_time = tf.shape(inputs)[1]

        # Reshape to 4D tensor `[B * T, input_size / 3, splice, 3(+Δ, ΔΔ)]`
        inputs = tf.reshape(
            inputs,
            shape=[batch_size * max_time, int(self.input_size / 3), 3, self.splice])
        inputs = tf.transpose(inputs, (0, 1, 3, 2))

        with tf.variable_scope('VGG1'):
            inputs = conv_layer(inputs,
                                filter_shape=[3, 3, 3, 64],
                                parameter_init=self.parameter_init,
                                relu=True,
                                name='conv1')
            inputs = conv_layer(inputs,
                                filter_shape=[3, 3, 64, 64],
                                parameter_init=self.parameter_init,
                                relu=True,
                                name='conv2')
            inputs = max_pool(inputs, name='max_pool')
            # TODO(hirofumi): try batch normalization

        with tf.variable_scope('VGG2'):
            inputs = conv_layer(inputs,
                                filter_shape=[3, 3, 64, 128],
                                parameter_init=self.parameter_init,
                                relu=True,
                                name='conv1')
            inputs = conv_layer(inputs,
                                filter_shape=[3, 3, 128, 128],
                                parameter_init=self.parameter_init,
                                relu=True,
                                name='conv2')
            inputs = max_pool(inputs, name='max_pool')
            # TODO(hirofumi): try batch normalization

        # Reshape to 2D tensor `[B * T, new_h * new_w * 128]`
        new_h = math.ceil(self.input_size / 3 / 4)  # expected to be 11 ro 10
        new_w = math.ceil(self.splice / 4)  # expected to be 3
        inputs = tf.reshape(
            inputs, shape=[batch_size * max_time, new_h * new_w * 128])

        # Insert linear layer to recude CNN's output demention
        # from (new_h * new_w * 128) to 256
        with tf.variable_scope('vgg_blsm_pipe') as scope:
            inputs = tf.contrib.layers.fully_connected(
                inputs=inputs,
                num_outputs=256,
                activation_fn=tf.nn.relu,
                scope=scope)

        # Dropout for the VGG-output-hidden connection
        inputs = tf.nn.dropout(inputs, keep_prob, name='dropout_pipe')

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
