#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Bidirectional LSTM encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class BLSTM_Encoder(object):
    """Bidirectional LSTM encoder.
    Args:
        num_unit: int, the number of units in each layer
        num_layer: int, the number of layers
        num_classes: int, the number of classes of target labels
            (except for a blank label)
        lstm_impl: string, BasicLSTMCell or LSTMCell or or LSTMBlockCell or
            LSTMBlockFusedCell.
            Choose the background implementation of tensorflow.
            Default is LSTMBlockCell (the fastest implementation).
        use_peephole: bool, if True, use peephole
        parameter_init: A float value. Range of uniform distribution to
            initialize weight parameters
        clip_activation: A float value. Range of activation clipping (> 0)
        num_proj: int, the number of nodes in recurrent projection layer
        bottleneck_dim: int, the dimensions of the bottleneck layer
        name: string, the name of encoder
    """

    def __init__(self,
                 num_unit,
                 num_layer,
                 num_classes,
                 lstm_impl='LSTMBlockCell',
                 use_peephole=True,
                 parameter_init=0.1,
                 clip_activation=None,
                 num_proj=None,
                 bottleneck_dim=None,
                 name='blstm_encoder'):

        self.num_unit = num_unit
        self.num_layer = num_layer
        self.num_classes = num_classes
        self.lstm_impl = lstm_impl
        self.use_peephole = use_peephole
        self.parameter_init = parameter_init
        self.clip_activation = clip_activation
        if lstm_impl != 'LSTMCell':
            self.num_proj = None
        elif num_proj not in [None, 0]:
            self.num_proj = int(num_proj)
        else:
            self.num_proj = None
        self.bottleneck_dim = int(bottleneck_dim) if bottleneck_dim not in [
            None, 0] else None
        self.name = name

    def __call__(self, inputs, inputs_seq_len, keep_prob_input,
                 keep_prob_hidden, keep_prob_output):
        """Construct model graph.
        Args:
            inputs: A tensor of size `[B, T, input_size]`
            inputs_seq_len: A tensor of size `[B]`
            keep_prob_input: A float value. A probability to keep nodes in
                the input-hidden connection
            keep_prob_hidden: A float value. A probability to keep nodes in
                the hidden-hidden connection
            keep_prob_output: A float value. A probability to keep nodes in
                the hidden-output connection
        Returns:
            logits: A tensor of size `[T, B, num_classes]`
            final_state: A final hidden state of the encoder
        """
        # Dropout for the input-hidden connection
        outputs = tf.nn.dropout(
            inputs, keep_prob_input, name='dropout_input')

        initializer = tf.random_uniform_initializer(
            minval=-self.parameter_init, maxval=self.parameter_init)

        # Hidden layers
        for i_layer in range(1, self.num_layer + 1, 1):
            with tf.variable_scope('blstm_hidden' + str(i_layer),
                                   initializer=initializer) as scope:
                if self.lstm_impl == 'BasicLSTMCell':
                    lstm_fw = tf.contrib.rnn.BasicLSTMCell(
                        self.num_unit,
                        forget_bias=1.0,
                        state_is_tuple=True,
                        activation=tf.tanh)
                    lstm_bw = tf.contrib.rnn.BasicLSTMCell(
                        self.num_unit,
                        forget_bias=1.0,
                        state_is_tuple=True,
                        activation=tf.tanh)

                elif self.lstm_impl == 'LSTMCell':
                    lstm_fw = tf.contrib.rnn.LSTMCell(
                        self.num_unit,
                        use_peepholes=self.use_peephole,
                        cell_clip=self.clip_activation,
                        num_proj=self.num_proj,
                        forget_bias=1.0,
                        state_is_tuple=True)
                    lstm_bw = tf.contrib.rnn.LSTMCell(
                        self.num_unit,
                        use_peepholes=self.use_peephole,
                        cell_clip=self.clip_activation,
                        num_proj=self.num_proj,
                        forget_bias=1.0,
                        state_is_tuple=True)

                elif self.lstm_impl == 'LSTMBlockCell':
                    # NOTE: This should be faster than tf.contrib.rnn.LSTMCell
                    lstm_fw = tf.contrib.rnn.LSTMBlockCell(
                        self.num_unit,
                        forget_bias=1.0,
                        # clip_cell=True,
                        use_peephole=self.use_peephole)
                    lstm_bw = tf.contrib.rnn.LSTMBlockCell(
                        self.num_unit,
                        forget_bias=1.0,
                        # clip_cell=True,
                        use_peephole=self.use_peephole)
                    # TODO: cell clipping (update for rc1.3)

                elif self.lstm_impl == 'LSTMBlockFusedCell':
                    raise NotImplementedError

                    # NOTE: This should be faster than
                    tf.contrib.rnn.LSTMBlockFusedCell
                    lstm_fw = tf.contrib.rnn.LSTMBlockFusedCell(
                        self.num_unit,
                        forget_bias=1.0,
                        # clip_cell=True,
                        use_peephole=self.use_peephole)
                    lstm_bw = tf.contrib.rnn.LSTMBlockFusedCell(
                        self.num_unit,
                        forget_bias=1.0,
                        # clip_cell=True,
                        use_peephole=self.use_peephole)
                    # TODO: cell clipping (update for rc1.3)

                else:
                    raise IndexError(
                        'lstm_impl is "BasicLSTMCell" or "LSTMCell" or "LSTMBlockCell" or "LSTMBlockFusedCell".')

                # Dropout for the hidden-hidden connections
                lstm_fw = tf.contrib.rnn.DropoutWrapper(
                    lstm_fw, output_keep_prob=keep_prob_hidden)
                lstm_bw = tf.contrib.rnn.DropoutWrapper(
                    lstm_bw, output_keep_prob=keep_prob_hidden)

                # _init_state_fw = lstm_fw.zero_state(self.batch_size,
                #                                     tf.float32)
                # _init_state_bw = lstm_bw.zero_state(self.batch_size,
                #                                     tf.float32)
                # initial_state_fw=_init_state_fw,
                # initial_state_bw=_init_state_bw,

                # Ignore 2nd return (the last state)
                (outputs_fw, outputs_bw), final_state = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=lstm_fw,
                    cell_bw=lstm_bw,
                    inputs=outputs,
                    sequence_length=inputs_seq_len,
                    dtype=tf.float32,
                    scope=scope)

                outputs = tf.concat(axis=2, values=[outputs_fw, outputs_bw])

        # Reshape to apply the same weights over the timesteps
        output_node = self.num_unit * 2 if self.num_proj is None else self.num_proj * 2
        outputs = tf.reshape(outputs, shape=[-1, output_node])

        # inputs: `[batch_size, max_time, input_size]`
        batch_size = tf.shape(inputs)[0]

        if self.bottleneck_dim is not None and self.bottleneck_dim != 0:
            with tf.name_scope('bottleneck'):
                # Affine
                W_bottleneck = tf.Variable(tf.truncated_normal(
                    shape=[output_node, self.bottleneck_dim],
                    stddev=0.1, name='W_bottleneck'))
                b_bottleneck = tf.Variable(tf.zeros(
                    shape=[self.bottleneck_dim], name='b_bottleneck'))
                outputs = tf.matmul(outputs, W_bottleneck) + b_bottleneck
                output_node = self.bottleneck_dim

                # Dropout for the hidden-output connections
                outputs = tf.nn.dropout(
                    outputs, keep_prob_output, name='dropout_output_bottle')

        with tf.name_scope('output'):
            # Affine
            W_output = tf.Variable(tf.truncated_normal(
                shape=[output_node, self.num_classes],
                stddev=0.1, name='W_output'))
            b_output = tf.Variable(tf.zeros(
                shape=[self.num_classes], name='b_output'))
            logits_2d = tf.matmul(outputs, W_output) + b_output

            # Reshape back to the original shape
            logits = tf.reshape(
                logits_2d, shape=[batch_size, -1, self.num_classes])

            # Convert to time-major: `[max_time, batch_size, num_classes]'
            logits = tf.transpose(logits, (1, 0, 2))

            # Dropout for the hidden-output connections
            logits = tf.nn.dropout(
                logits, keep_prob_output, name='dropout_output')
            # NOTE: This may lead to bad results

            return logits, final_state
