#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Bidirectional GRU Encoder class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .encoder_base import EncoderOutput, EncoderBase


class BGRUEncoder(EncoderBase):
    """Bidirectional GRU Encoder.
    Args:
        num_units:
        num_layer:
        keep_prob_input:
        keep_prob_hidden:
        parameter_init:
        clip_activation: not used
        num_proj: not used
    """

    def __init__(self,
                 num_units,
                 num_layer,
                 keep_prob_input=1.0,
                 keep_prob_hidden=1.0,
                 parameter_init=0.1,
                 clip_activation=50,  # not used
                 num_proj=None,  # not used
                 name='bgru_encoder'):

        EncoderBase.__init__(self, num_units, num_layer, keep_prob_input,
                             keep_prob_hidden, parameter_init, clip_activation,
                             num_proj, name)

    def _build(self, inputs, inputs_seq_len):
        """Construct Bidirectional GRU encoder.
        Args:
            inputs:
            inputs_seq_len:
        Returns:
            EncoderOutput: A tuple of
                `(outputs, final_state,
                        attention_values, attention_values_length)`
                outputs:
                final_state:
                attention_values:
                attention_values_length:
        """
        self.inputs = inputs
        self.inputs_seq_len = inputs_seq_len

        # Input dropout
        outputs = tf.nn.dropout(inputs,
                                self.keep_prob_input,
                                name='dropout_input')

        # Hidden layers
        for i_layer in range(self.num_layer):
            with tf.name_scope('BiGRU_encoder_hidden' + str(i_layer + 1)):

                initializer = tf.random_uniform_initializer(
                    minval=-self.parameter_init,
                    maxval=self.parameter_init)

                with tf.variable_scope('GRU', initializer=initializer):
                    gru_fw = tf.contrib.rnn.GRUCell(self.num_units)
                    gru_bw = tf.contrib.rnn.GRUCell(self.num_units)

                # Dropout (output)
                gru_fw = tf.contrib.rnn.DropoutWrapper(
                    gru_fw,
                    output_keep_prob=self.keep_prob_hidden)
                gru_bw = tf.contrib.rnn.DropoutWrapper(
                    gru_bw,
                    output_keep_prob=self.keep_prob_hidden)

                # _init_state_fw = lstm_fw.zero_state(self.batch_size,
                #                                     tf.float32)
                # _init_state_bw = lstm_bw.zero_state(self.batch_size,
                #                                     tf.float32)
                # initial_state_fw=_init_state_fw,
                # initial_state_bw=_init_state_bw,

                (outputs_fw, outputs_bw), final_state = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=gru_fw,
                    cell_bw=gru_bw,
                    inputs=outputs,
                    sequence_length=inputs_seq_len,
                    dtype=tf.float32,
                    scope='BiGRU_' + str(i_layer + 1))

                outputs = tf.concat(
                    axis=2, values=[outputs_fw, outputs_bw])

        return EncoderOutput(outputs=outputs,
                             final_state=final_state,
                             attention_values=outputs,
                             attention_values_length=inputs_seq_len)
