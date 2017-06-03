#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Batch Normalized LSTM-CTC model."""

import tensorflow as tf
from .ctc_base import ctcBase

from recurrent.layers.bn_lstm import BatchNormLSTMCell
from recurrent.initializer import orthogonal_initializer


class BN_LSTM_CTC(ctcBase):
    """Batch Normalized LSTM-CTC model.
    Args:
        batch_size: int, batch size of mini batch
        input_size: int, the dimension of input vectors
        num_cell: int, the number of memory cells in each layer
        num_layers: int, the number of layers
        output_size: int, the number of nodes in softmax layer (except for blank class)
        parameter_init: A float value. Range of uniform distribution to initialize weight parameters
        clip_grad: A float value. Range of gradient clipping (non-negative)
        clip_activation: A float value. Range of activation clipping (non-negative)
        dropout_ratio_input: A float value. Dropout ratio in input-hidden layers
        dropout_ratio_hidden: A float value. Dropout ratio in hidden-hidden layers
        is_training: bool, set True when training.
    """

    def __init__(self,
                 batch_size,
                 input_size,
                 num_cell,
                 num_layers,
                 output_size,
                 parameter_init=0.1,
                 clip_grad=None,
                 clip_activation=None,
                 dropout_ratio_input=1.0,
                 dropout_ratio_hidden=1.0,
                 is_training=True):

        ctcBase.__init__(self, batch_size, input_size, num_cell, num_layers,
                         output_size, parameter_init, clip_grad, clip_activation,
                         dropout_ratio_input, dropout_ratio_hidden)

        self._is_training = is_training

        # Define model graph
        self._build()

    def _build(self):
        """Construct network."""
        self.is_training_pl = tf.placeholder(tf.bool)

        lstm_list = []
        for i_layer in range(self.num_layers):
            with tf.name_scope('LSTM_hidden' + str(i_layer + 1)):

                initializer = tf.random_uniform_initializer(minval=-self.parameter_init,
                                                            maxval=self.parameter_init)
                # initializer = orthogonal_initializer()

                lstm = BatchNormLSTMCell(self.num_cell,
                                         use_peepholes=True,
                                         cell_clip=self.clip_activation,
                                         initializer=initializer,
                                         forget_bias=1.0,
                                         state_is_tuple=True,
                                         is_training=self.is_training_pl)

                # Dropout (output)
                lstm = tf.contrib.rnn.DropoutWrapper(
                    lstm, output_keep_prob=self.keep_prob_hidden_pl)

        # Stack multiple cells
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            lstm_list, state_is_tuple=True)

        # Ignore 2nd return (the last state)
        outputs, _ = tf.nn.dynamic_rnn(cell=stacked_lstm,
                                       inputs=self.inputs,
                                       sequence_length=self.seq_len_pl,
                                       dtype=tf.float32)

        with tf.name_scope('output'):
            # (batch_size, max_timesteps, input_size_splice)
            inputs_shape = tf.shape(self.inputs_pl)
            batch_size, max_timesteps = inputs_shape[0], inputs_shape[1]

            # Reshape to apply the same weights over the timesteps
            outputs = tf.reshape(outputs, shape=[-1, self.num_cell])

            # Affine
            W_output = tf.Variable(tf.truncated_normal(shape=[self.num_cell, self.num_classes],
                                                       stddev=0.1, name='W_output'))
            b_output = tf.Variable(
                tf.zeros(shape=[self.num_classes], name='b_output'))
            logits_2d = tf.matmul(outputs, W_output) + b_output

            # Reshape back to the original shape
            logits_3d = tf.reshape(
                logits_2d, shape=[batch_size, -1, self.num_classes])

            # Convert to (max_timesteps, batch_size, num_classes)
            self.logits = tf.transpose(logits_3d, (1, 0, 2))
