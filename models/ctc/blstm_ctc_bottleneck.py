#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Bidirectional LSTM-CTC model with a bottleneck layer."""

import tensorflow as tf
from .ctc_base import ctcBase


class BLSTM_CTC_BOTTLENECK(ctcBase):
    """Bidirectional LSTM-CTC model with a bottleneck layer.
    Args:
        batch_size: int, batch size of mini batch
        input_size: int, the dimensions of input vectors
        num_cell: int, the number of memory cells in each layer
        num_layer: int, the number of layers
        bottleneck_dim: int, the dimensions of the bottleneck layer
        output_size: int, the number of nodes in softmax layer (except for blank class)
        parameter_init: A float value. Range of uniform distribution to initialize weight parameters
        clip_gradients: A float value. Range of gradient clipping (non-negative)
        clip_activation: A float value. Range of activation clipping (non-negative)
        dropout_ratio_input: A float value. Dropout ratio in input-hidden layers
        dropout_ratio_hidden: A float value. Dropout ratio in hidden-hidden layers
        num_proj: int, the number of nodes in recurrent projection layer
        weight_decay: A float value. Regularization parameter for weight decay
    """

    def __init__(self,
                 batch_size,
                 input_size,
                 num_cell,
                 num_layer,
                 bottleneck_dim,
                 output_size,
                 parameter_init=0.1,
                 clip_gradients=None,
                 clip_activation=None,
                 dropout_ratio_input=1.0,
                 dropout_ratio_hidden=1.0,
                 num_proj=None,
                 weight_decay=0.0):

        ctcBase.__init__(self, batch_size, input_size, num_cell, num_layer,
                         output_size, parameter_init,
                         clip_gradients, clip_activation,
                         dropout_ratio_input, dropout_ratio_hidden,
                         weight_decay)

        self.bottleneck_dim = bottleneck_dim
        self.num_proj = None if num_proj == 0 else num_proj

    def define(self):
        """Construct Bidirectional LSTM layers."""
        # Generate placeholders
        self._generate_pl()

        # Dropout for Input
        self.inputs = tf.nn.dropout(self.inputs_pl,
                                    self.keep_prob_input_pl,
                                    name='dropout_input')

        # Hidden layers
        outputs = self.inputs
        for i_layer in range(self.num_layer):
            with tf.name_scope('BiLSTM_hidden' + str(i_layer + 1)):

                initializer = tf.random_uniform_initializer(minval=-self.parameter_init,
                                                            maxval=self.parameter_init)

                lstm_fw = tf.contrib.rnn.LSTMCell(self.num_cell,
                                                  use_peepholes=True,
                                                  cell_clip=self.clip_activation,
                                                  initializer=initializer,
                                                  num_proj=self.num_proj,
                                                  forget_bias=1.0,
                                                  state_is_tuple=True)
                lstm_bw = tf.contrib.rnn.LSTMCell(self.num_cell,
                                                  use_peepholes=True,
                                                  cell_clip=self.clip_activation,
                                                  initializer=initializer,
                                                  num_proj=self.num_proj,
                                                  forget_bias=1.0,
                                                  state_is_tuple=True)

                # Dropout (output)
                lstm_fw = tf.contrib.rnn.DropoutWrapper(lstm_fw,
                                                        output_keep_prob=self.keep_prob_hidden_pl)
                lstm_bw = tf.contrib.rnn.DropoutWrapper(lstm_bw,
                                                        output_keep_prob=self.keep_prob_hidden_pl)

                # _init_state_fw = lstm_fw.zero_state(self.batch_size, tf.float32)
                # _init_state_bw = lstm_bw.zero_state(self.batch_size, tf.float32)
                # initial_state_fw=_init_state_fw,
                # initial_state_bw=_init_state_bw,

                # Ignore 2nd return (the last state)
                (outputs_fw, outputs_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=lstm_fw,
                    cell_bw=lstm_bw,
                    inputs=outputs,
                    sequence_length=self.seq_len_pl,
                    dtype=tf.float32,
                    scope='BiLSTM_' + str(i_layer + 1))

                outputs = tf.concat(axis=2, values=[outputs_fw, outputs_bw])

        # Reshape to apply the same weights over the timesteps
        if self.num_proj is None:
            output_node = self.num_cell * 2
        else:
            output_node = self.num_proj * 2
        outputs = tf.reshape(outputs, shape=[-1, output_node])

        # (batch_size, max_timesteps, input_size_splice)
        inputs_shape = tf.shape(self.inputs_pl)
        batch_size, max_timesteps = inputs_shape[0], inputs_shape[1]

        with tf.name_scope('bottleneck'):
            # Affine
            W_bottleneck = tf.Variable(tf.truncated_normal(shape=[output_node, self.bottleneck_dim],
                                                           stddev=0.1, name='W_bottleneck'))
            b_bottleneck = tf.Variable(
                tf.zeros(shape=[self.bottleneck_dim], name='b_bottleneck'))
            logits_2d = tf.matmul(outputs, W_bottleneck) + b_bottleneck

        with tf.name_scope('output'):
            # Affine
            W_output = tf.Variable(tf.truncated_normal(shape=[self.bottleneck_dim, self.num_classes],
                                                       stddev=0.1, name='W_output'))
            b_output = tf.Variable(
                tf.zeros(shape=[self.num_classes], name='b_output'))
            logits_2d = tf.matmul(logits_2d, W_output) + b_output

            # Reshape back to the original shape
            logits_3d = tf.reshape(
                logits_2d, shape=[batch_size, -1, self.num_classes])

            # Convert to (max_timesteps, batch_size, num_classes)
            self.logits = tf.transpose(logits_3d, (1, 0, 2))
