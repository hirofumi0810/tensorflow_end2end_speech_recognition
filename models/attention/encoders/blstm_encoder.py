#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Bidirectional LSTM Encoder class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from models.attention.encoders.encoder_base import EncoderOutput, EncoderBase


class BLSTMEncoder(EncoderBase):
    """Bidirectional LSTM Encoder.
    Args:
        num_unit: int, the number of units in each layer
        num_layer: int, the number of layers
        parameter_init: A float value. Range of uniform distribution to
            initialize weight parameters
        clip_activation: A float value. Range of activation clipping (> 0)
        num_proj: int, the number of nodes in recurrent projection layer
    """

    def __init__(self,
                 num_unit,
                 num_layer,
                 parameter_init=0.1,
                 clip_activation=50,
                 num_proj=None,
                 name='blstm_encoder'):

        EncoderBase.__init__(self, num_unit, num_layer,
                             parameter_init, clip_activation,
                             num_proj, name)

    def _build(self, inputs, inputs_seq_len,
               keep_prob_input, keep_prob_hidden):
        """Construct Bidirectional LSTM encoder.
        Args:
            inputs: A tensor of `[batch_size, time, input_dim]`
            inputs_seq_len: A tensor of `[batch_size]`
            keep_prob_input: A float value. A probability to keep nodes in
                the input-hidden layer
            keep_prob_hidden: A float value. A probability to keep nodes in
                the hidden-hidden layers
        Returns:
            EncoderOutput: A namedtaple of
                `(outputs, final_state,
                        attention_values, attention_values_length)`
                outputs:
                final_state: LSTMStateTuple
                attention_values:
                attention_values_length:
        """
        self.inputs = inputs
        self.inputs_seq_len = inputs_seq_len

        # Input dropout
        outputs = tf.nn.dropout(inputs,
                                keep_prob_input,
                                name='dropout_input')

        # Hidden layers
        for i_layer in range(self.num_layer):
            with tf.name_scope('blstm_encoder_hidden' + str(i_layer + 1)):

                initializer = tf.random_uniform_initializer(
                    minval=-self.parameter_init,
                    maxval=self.parameter_init)

                lstm_fw = tf.contrib.rnn.LSTMCell(
                    self.num_unit,
                    use_peepholes=True,
                    cell_clip=self.clip_activation,
                    initializer=initializer,
                    num_proj=None,
                    forget_bias=1.0,
                    state_is_tuple=True)
                lstm_bw = tf.contrib.rnn.LSTMCell(
                    self.num_unit,
                    use_peepholes=True,
                    cell_clip=self.clip_activation,
                    initializer=initializer,
                    num_proj=self.num_proj,
                    forget_bias=1.0,
                    state_is_tuple=True)

                # Dropout (output)
                lstm_fw = tf.contrib.rnn.DropoutWrapper(
                    lstm_fw,
                    output_keep_prob=keep_prob_hidden)
                lstm_bw = tf.contrib.rnn.DropoutWrapper(
                    lstm_bw,
                    output_keep_prob=keep_prob_hidden)

                # _init_state_fw = lstm_fw.zero_state(self.batch_size,
                #                                     tf.float32)
                # _init_state_bw = lstm_bw.zero_state(self.batch_size,
                #                                     tf.float32)
                # initial_state_fw=_init_state_fw,
                # initial_state_bw=_init_state_bw,

                # Stacking
                (outputs_fw, outputs_bw), final_state = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=lstm_fw,
                    cell_bw=lstm_bw,
                    inputs=outputs,
                    sequence_length=inputs_seq_len,
                    dtype=tf.float32,
                    scope='blstm_dynamic' + str(i_layer + 1))

                # Concatenate each direction
                outputs = tf.concat(axis=2, values=[outputs_fw, outputs_bw])

        return EncoderOutput(outputs=outputs,
                             final_state=final_state,
                             attention_values=outputs,
                             attention_values_length=inputs_seq_len)
