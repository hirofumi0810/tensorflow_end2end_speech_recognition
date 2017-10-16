#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Unidirectional LSTM encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class LSTMEncoder(object):
    """Unidirectional LSTM encoder.
    Args:
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
                 num_units,
                 num_proj,
                 num_layers,
                 lstm_impl,
                 use_peephole,
                 parameter_init,
                 clip_activation,
                 time_major=False,
                 name='lstm_encoder'):
        if num_proj == 0:
            raise ValueError

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
            outputs: Encoder states.
                if time_major is True, a tensor of size
                    `[T, B, num_units (num_proj)]`
                otherwise, `[B, T, num_units (num_proj)]`
            final_state: A final hidden state of the encoder
        """
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
                inputs, inputs_seq_len, keep_prob, initializer,
                self.time_major)

        elif self.lstm_impl == 'CudnnLSTM':
            outputs, final_state = cudnnlstm(
                self.num_units, self.num_layers,
                inputs, inputs_seq_len, keep_prob, initializer,
                self.time_major)
        else:
            raise IndexError(
                'lstm_impl is "BasicLSTMCell" or "LSTMCell" or ' +
                '"LSTMBlockCell" or "LSTMBlockFusedCell" or ' +
                '"CudnnLSTM".')

        return outputs, final_state


def basiclstmcell(num_units, num_layers, inputs, inputs_seq_len,
                  keep_prob, initializer, time_major, num_layers_sub=None):

    if time_major:
        # Convert form batch-major to time-major
        inputs = tf.transpose(inputs, [1, 0, 2])

    lstm_list = []
    with tf.variable_scope('multi_lstm', initializer=initializer) as scope:
        for i_layer in range(1, num_layers + 1, 1):

            lstm = tf.contrib.rnn.BasicLSTMCell(
                num_units,
                forget_bias=1.0,
                state_is_tuple=True,
                activation=tf.tanh)

            # Dropout for the hidden-hidden connections
            lstm = tf.contrib.rnn.DropoutWrapper(
                lstm, output_keep_prob=keep_prob)

            lstm_list.append(lstm)

            if num_layers_sub is not None and i_layer == num_layers_sub:
                lstm_list_sub = lstm_list

        # Stack multiple cells
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            lstm_list, state_is_tuple=True)

        # Ignore 2nd return (the last state)
        outputs, final_state = tf.nn.dynamic_rnn(
            cell=stacked_lstm,
            inputs=inputs,
            sequence_length=inputs_seq_len,
            dtype=tf.float32,
            time_major=time_major,
            scope=scope)
        # NOTE: initial states are zero states by default

    if num_layers_sub is not None:
        with tf.variable_scope('multi_lstm', initializer=initializer, reuse=True) as scope:
            # Stack multiple cells
            stacked_lstm_sub = tf.contrib.rnn.MultiRNNCell(
                lstm_list_sub, state_is_tuple=True)

            # Ignore 2nd return (the last state)
            outputs_sub, final_state_sub = tf.nn.dynamic_rnn(
                cell=stacked_lstm_sub,
                inputs=inputs,
                sequence_length=inputs_seq_len,
                dtype=tf.float32,
                time_major=time_major,
                scope=scope)
        return outputs, final_state, outputs_sub, final_state_sub
    else:
        return outputs, final_state


def lstmcell(num_units, num_proj, num_layers, use_peephole, clip_activation,
             inputs, inputs_seq_len, keep_prob, initializer, time_major,
             num_layers_sub=None):

    if time_major:
        # Convert form batch-major to time-major
        inputs = tf.transpose(inputs, [1, 0, 2])

    lstm_list = []
    with tf.variable_scope('multi_lstm', initializer=initializer) as scope:
        for i_layer in range(1, num_layers + 1, 1):

            lstm = tf.contrib.rnn.LSTMCell(
                num_units,
                use_peepholes=use_peephole,
                cell_clip=clip_activation,
                num_proj=num_proj,
                forget_bias=1.0,
                state_is_tuple=True)

            # Dropout for the hidden-hidden connections
            lstm = tf.contrib.rnn.DropoutWrapper(
                lstm, output_keep_prob=keep_prob)

            lstm_list.append(lstm)

            if num_layers_sub is not None and i_layer == num_layers_sub:
                lstm_list_sub = lstm_list

        # Stack multiple cells
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            lstm_list, state_is_tuple=True)

        # Ignore 2nd return (the last state)
        outputs, final_state = tf.nn.dynamic_rnn(
            cell=stacked_lstm,
            inputs=inputs,
            sequence_length=inputs_seq_len,
            dtype=tf.float32,
            time_major=time_major,
            scope=scope)
        # NOTE: initial states are zero states by default

    if num_layers_sub is not None:
        with tf.variable_scope('multi_lstm', initializer=initializer, reuse=True) as scope:
            # Stack multiple cells
            stacked_lstm_sub = tf.contrib.rnn.MultiRNNCell(
                lstm_list_sub, state_is_tuple=True)

            # Ignore 2nd return (the last state)
            outputs_sub, final_state_sub = tf.nn.dynamic_rnn(
                cell=stacked_lstm_sub,
                inputs=inputs,
                sequence_length=inputs_seq_len,
                dtype=tf.float32,
                time_major=time_major,
                scope=scope)
        return outputs, final_state, outputs_sub, final_state_sub
    else:
        return outputs, final_state


def lstmblockcell(num_units, num_layers, use_peephole, clip_activation, inputs,
                  inputs_seq_len, keep_prob, initializer, time_major,
                  num_layers_sub=None):

    if time_major:
        # Convert form batch-major to time-major
        inputs = tf.transpose(inputs, [1, 0, 2])

    lstm_list = []
    with tf.variable_scope('multi_lstm', initializer=initializer) as scope:
        for i_layer in range(1, num_layers + 1, 1):

            if tf.__version__ == '1.3.0':
                lstm = tf.contrib.rnn.LSTMBlockCell(
                    num_units,
                    forget_bias=1.0,
                    clip_cell=clip_activation,
                    use_peephole=use_peephole)
            else:
                lstm = tf.contrib.rnn.LSTMBlockCell(
                    num_units,
                    forget_bias=1.0,
                    use_peephole=use_peephole)

            # Dropout for the hidden-hidden connections
            lstm = tf.contrib.rnn.DropoutWrapper(
                lstm, output_keep_prob=keep_prob)

            lstm_list.append(lstm)

            if num_layers_sub is not None and i_layer == num_layers_sub:
                lstm_list_sub = lstm_list

        # Stack multiple cells
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            lstm_list, state_is_tuple=True)

        # Ignore 2nd return (the last state)
        outputs, final_state = tf.nn.dynamic_rnn(
            cell=stacked_lstm,
            inputs=inputs,
            sequence_length=inputs_seq_len,
            dtype=tf.float32,
            time_major=time_major,
            scope=scope)
        # NOTE: initial states are zero states by default

    if num_layers_sub is not None:
        with tf.variable_scope('multi_lstm', initializer=initializer, reuse=True) as scope:
            # Stack multiple cells
            stacked_lstm_sub = tf.contrib.rnn.MultiRNNCell(
                lstm_list_sub, state_is_tuple=True)

            # Ignore 2nd return (the last state)
            outputs_sub, final_state_sub = tf.nn.dynamic_rnn(
                cell=stacked_lstm_sub,
                inputs=inputs,
                sequence_length=inputs_seq_len,
                dtype=tf.float32,
                time_major=time_major,
                scope=scope)
        return outputs, final_state, outputs_sub, final_state_sub
    else:
        return outputs, final_state


def lstmblockfusedcell(num_units, num_layers, use_peephole, clip_activation,
                       inputs, inputs_seq_len, keep_prob, initializer,
                       time_major, num_layers_sub=None):
    raise NotImplementedError


def cudnnlstm(num_units, num_layers, parameter_init,
              inputs, inputs_seq_len, keep_prob, initializer, time_major,
              num_layers_sub=None):
    raise NotImplementedError
