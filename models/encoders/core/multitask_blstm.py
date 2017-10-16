#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Multi-task bidirectional LSTM encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from models.encoders.core.blstm import basiclstmcell, lstmcell, lstmblockcell, lstmblockfusedcell, cudnnlstm


class MultitaskBLSTMEncoder(object):
    """Multi-task bidirectional LSTM encoder.
    Args:
        num_units (int): the number of units in each layer
        num_proj (int): the number of nodes in recurrent projection layer
        num_layers_main (int): the number of layers of the main task
        num_layers_sub (int): the number of layers of the sub task
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
                 num_layers_main,
                 num_layers_sub,
                 lstm_impl,
                 use_peephole,
                 parameter_init,
                 clip_activation,
                 time_major=False,
                 name='multitask_blstm_encoder'):
        if num_proj == 0:
            raise ValueError

        self.num_units = num_units
        if lstm_impl != 'LSTMCell':
            self.num_proj = None
        else:
            self.num_proj = num_proj
        # TODO: fix this
        self.num_layers_main = num_layers_main
        self.num_layers_sub = num_layers_sub
        self.lstm_impl = lstm_impl
        self.use_peephole = use_peephole
        self.parameter_init = parameter_init
        self.clip_activation = clip_activation
        self.time_major = time_major
        self.name = name

        if self.num_layers_sub < 1 or self.num_layers_main < self.num_layers_sub:
            raise ValueError(
                'Set num_layers_sub between 1 to num_layers_main.')

    def __call__(self, inputs, inputs_seq_len, keep_prob):
        """Construct model graph.
        Args:
            inputs (placeholder): A tensor of size`[B, T, input_size]`
            inputs_seq_len (placeholder): A tensor of size` [B]`
            keep_prob (placeholder, float): A probability to keep nodes
                in the hidden-hidden connection
        Returns:
            outputs: A tensor of size `[T, B, input_size]` in the main task
            final_state: A final hidden state of the encoder in the main task
            outputs_sub: A tensor of size `[T, B, input_size]` in the sub task
            final_state_sub: A final hidden state of the encoder in the sub task
        """
        initializer = tf.random_uniform_initializer(
            minval=-self.parameter_init, maxval=self.parameter_init)

        if self.lstm_impl == 'BasicLSTMCell':
            outputs, final_state, outputs_sub, final_state_sub = basiclstmcell(
                self.num_units, self.num_layers_main,
                inputs, inputs_seq_len, keep_prob, initializer,
                self.time_major, self.num_layers_sub)

        elif self.lstm_impl == 'LSTMCell':
            outputs, final_state, outputs_sub, final_state_sub = lstmcell(
                self.num_units, self.num_proj, self.num_layers_main,
                self.use_peephole, self.clip_activation,
                inputs, inputs_seq_len, keep_prob, initializer,
                self.time_major, self.num_layers_sub)

        elif self.lstm_impl == 'LSTMBlockCell':
            outputs, final_state, outputs_sub, final_state_sub = lstmblockcell(
                self.num_units, self.num_layers_main,
                self.use_peephole, self.clip_activation,
                inputs, inputs_seq_len, keep_prob, initializer,
                self.time_major, self.num_layers_sub)

        elif self.lstm_impl == 'LSTMBlockFusedCell':
            outputs, final_state, outputs_sub, final_state_sub = lstmblockfusedcell(
                self.num_units, self.num_layers_main,
                inputs, inputs_seq_len, keep_prob, initializer,
                self.time_major, self.num_layers_sub)

        elif self.lstm_impl == 'CudnnLSTM':
            outputs, final_state, outputs_sub, final_state_sub = cudnnlstm(
                self.num_units, self.num_layers_main,
                inputs, inputs_seq_len, keep_prob, initializer,
                self.time_major, self.num_layers_sub)

        else:
            raise IndexError(
                'lstm_impl is "BasicLSTMCell" or "LSTMCell" or ' +
                '"LSTMBlockCell" or "LSTMBlockFusedCell" or ' +
                '"CudnnLSTM".')

        return outputs, final_state, outputs_sub, final_state_sub
