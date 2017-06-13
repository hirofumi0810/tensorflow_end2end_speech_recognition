#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""GRU Encoder class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .encoder_base import EncoderOutput, EncoderBase


class GRUEncoder(EncoderBase):
    """GRU Encoder.
    Args:
        num_unit:
        num_layer:
        keep_prob_input:
        keep_prob_hidden:
        parameter_init:
        clip_activation: not used
        num_proj: not used
    """

    def __init__(self,
                 num_unit,
                 num_layer,
                 keep_prob_input=1.0,
                 keep_prob_hidden=1.0,
                 parameter_init=0.1,
                 clip_activation=50,  # not used
                 num_proj=None,  # not used
                 name='gru_encoder'):

        EncoderBase.__init__(self, num_unit, num_layer, keep_prob_input,
                             keep_prob_hidden, parameter_init, clip_activation,
                             num_proj, name)

    def _build(self, inputs, inputs_seq_len):
        """Construct GRU encoder.
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
        inputs = tf.nn.dropout(inputs,
                               self.keep_prob_input,
                               name='dropout_input')

        # Hidden layers
        gru_list = []
        for i_layer in range(self.num_layer):
            with tf.name_scope('GRU_encoder_hidden' + str(i_layer + 1)):

                initializer = tf.random_uniform_initializer(
                    minval=-self.parameter_init,
                    maxval=self.parameter_init)

                with tf.variable_scope('GRU', initializer=initializer):
                    gru = tf.contrib.rnn.GRUCell(self.num_unit)

                # Dropout (output)
                gru = tf.contrib.rnn.DropoutWrapper(
                    gru, output_keep_prob=self.keep_prob_hidden)

                gru_list.append(gru)

        # Stack multiple cells
        stacked_gru = tf.contrib.rnn.MultiRNNCell(
            gru_list, state_is_tuple=True)

        outputs, final_state = tf.nn.dynamic_rnn(cell=stacked_gru,
                                                 inputs=inputs,
                                                 sequence_length=inputs_seq_len,
                                                 dtype=tf.float32)

        return EncoderOutput(outputs=outputs,
                             final_state=final_state,
                             attention_values=outputs,
                             attention_values_length=inputs_seq_len)
