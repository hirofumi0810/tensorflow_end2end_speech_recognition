#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Pyramidal Bidirectional LSTM Encoder class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .encoder_base import EncoderOutput, EncoderBase


class PyramidalBLSTMEncoder(EncoderBase):
    """Pyramidal Bidirectional LSTM Encoder.
    Args:
        num_unit:
        num_layer:
        keep_prob_input:
        keep_prob_hidden:
        parameter_init:
        clip_activation:
        num_proj:
    """

    def __init__(self,
                 num_unit,
                 num_layer,
                 keep_prob_input=1.0,
                 keep_prob_hidden=1.0,
                 parameter_init=0.1,
                 clip_activation=50,
                 num_proj=None,
                 name='pblstm_encoder'):

        if num_unit % 2 != 0:
            raise ValueError('num_unit should be even number.')

        EncoderBase.__init__(self, num_unit, num_layer, keep_prob_input,
                             keep_prob_hidden, parameter_init, clip_activation,
                             num_proj, name)

    def _build(self, inputs, inputs_seq_len):
        """Construct Pyramidal Bidirectional LSTM encoder.
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
            with tf.name_scope('Pyramidal_BiLSTM_encoder_hidden' + str(i_layer + 1)):

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
                    output_keep_prob=self.keep_prob_hidden)
                lstm_bw = tf.contrib.rnn.DropoutWrapper(
                    lstm_bw,
                    output_keep_prob=self.keep_prob_hidden)

                # _init_state_fw = lstm_fw.zero_state(self.batch_size,
                #                                     tf.float32)
                # _init_state_bw = lstm_bw.zero_state(self.batch_size,
                #                                     tf.float32)
                # initial_state_fw=_init_state_fw,
                # initial_state_bw=_init_state_bw,

                # Convert to `[max_time, batch_size, input_size]`
                outputs = tf.transpose(outputs, (1, 0, 2))
                max_time = outputs.get_shape()[0]

                # Concatenate each 2 time steps to reduce time resolution
                def concat_fn(inputs, current_time):
                    if current_time % 2 != 0:
                        concat_input = tf.concat(
                            axis=0,
                            values=[outputs[current_time - 1, :, :],
                                    outputs[current_time, :, :]],
                            name='pblstm_concat')
                        return concat_input

                # Use tf.map_fn to apply concat_fn to each tensor in outputs, along
                # dimension 0 (timestep dimension)
                # concat_list = tf.map_fn(concat_fn, outputs.value_index)
                # concat_list = tf.foldl(concat_fn, outputs.value_index)
                concat_list = tf.while_loop(
                    cond=lambda x: tf.less(x, max_time),
                    body=concat_fn,
                    loop_vars=[outputs.value_index])

                outputs = tf.pack(concat_list, axis=0, name='pblstm_pack')

                # Reshape to `[batch_size, max_time, input_size]`
                outputs = tf.transpose(outputs, (1, 0, 2))

                # Stacking
                (outputs_fw, outputs_bw), final_state = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=lstm_fw,
                    cell_bw=lstm_bw,
                    inputs=outputs,
                    sequence_length=inputs_seq_len,
                    dtype=tf.float32,
                    scope='Pyramidal_BiLSTM_' + str(i_layer + 1))

                # Concatenate each direction
                outputs = tf.concat(axis=2, values=[outputs_fw, outputs_bw])

        return EncoderOutput(outputs=outputs,
                             final_state=final_state,
                             attention_values=outputs,
                             attention_values_length=inputs_seq_len)
