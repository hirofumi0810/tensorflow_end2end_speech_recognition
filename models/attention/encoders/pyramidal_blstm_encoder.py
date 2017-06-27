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
        num_unit: int, the number of units in each layer
        num_layer: int, the number of layers
        parameter_init: A float value. Range of uniform distribution to
            initialize weight parameters
        clip_activation: A float value. Range of activation clipping (> 0)
        num_proj: not used
    """

    def __init__(self,
                 num_unit,
                 num_layer,
                 parameter_init=0.1,
                 clip_activation=50,
                 num_proj=None,
                 concat=False,
                 name='pblstm_encoder'):

        # if num_unit % 2 != 0:
        #     raise ValueError('num_unit should be even number.')

        EncoderBase.__init__(self, num_unit, num_layer,
                             parameter_init, clip_activation,
                             num_proj, name)

    def _build(self, inputs, inputs_seq_len,
               keep_prob_input, keep_prob_hidden):
        """Construct Pyramidal Bidirectional LSTM encoder.
        Args:
            inputs: A tensor of `[batch_size, time, input_dim]`
            inputs_seq_len: A tensor of `[batch_size]`
            keep_prob_input: A float value. A probability to keep nodes in
                input-hidden layers
            keep_prob_hidden: A float value. A probability to keep nodes in
                hidden-hidden layers
        Returns:
            EncoderOutput: A namedtuple of
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
                                keep_prob_input,
                                name='dropout_input')

        # Hidden layers
        for i_layer in range(self.num_layer):
            with tf.name_scope('pblstm_encoder_hidden' + str(i_layer + 1)):

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

                if i_layer > 0:
                    # Convert to `[time, batch_size, input_size]`
                    outputs = tf.transpose(outputs, (1, 0, 2))
                    max_time = tf.shape(outputs)[0]
                    batch_size = tf.shape(outputs)[1]
                    feature_dim = outputs.get_shape().as_list()[2]

                    max_time_half = tf.floor(max_time / 2) + 1

                    # Apply concat_fn to each tensor in outputs along
                    # dimension 0 (times-axis)
                    i_time = tf.constant(0)
                    final_time, outputs, tensor_list = tf.while_loop(
                        cond=lambda t, hidden, tensor_list: t < max_time,
                        body=lambda t, hidden, tensor_list: self._concat_fn(
                            t, hidden, tensor_list),
                        loop_vars=[i_time, outputs, tf.Variable([])],
                        shape_invariants=[i_time.get_shape(),
                                          outputs.get_shape(),
                                          tf.TensorShape([None])])

                    outputs = tf.stack(tensor_list, axis=0)

                    inputs_seq_len = tf.cast(tf.floor(
                        tf.cast(inputs_seq_len, tf.float32) / 2),
                        tf.int32)

                    # Transpose to `[batch_size, time, input_size]`
                    outputs = tf.transpose(outputs, (1, 0, 2))

                # Stacking
                (outputs_fw, outputs_bw), final_state = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=lstm_fw,
                    cell_bw=lstm_bw,
                    inputs=outputs,
                    sequence_length=inputs_seq_len,
                    dtype=tf.float32,
                    scope='pblstm_dynamic_' + str(i_layer + 1))

                # Concatenate each direction
                outputs = tf.concat(axis=2, values=[outputs_fw, outputs_bw])

        return EncoderOutput(outputs=outputs,
                             final_state=final_state,
                             attention_values=outputs,
                             attention_values_length=inputs_seq_len)

    def _concat_fn(self, current_time, x, tensor_list):
        """Concatenate each 2 time steps to reduce time resolution.
        Args:
            current_time: The current timestep
            x: A tensor of size `[max_time, batch_size, feature_dim]`
            result: A tensor of size `[t, batch_size, feature_dim * 2]`
        Returns:
            current_time: current_time + 2
            x: A tensor of size `[max_time, batch_size, feature_dim]`
            result: A tensor of size `[t + 1, batch_size, feature_dim * 2]`
        """
        print(tensor_list)
        print(current_time)
        print('-----')

        batch_size = tf.shape(x)[1]
        feature_dim = x.get_shape().as_list()[2]

        # Concat features in 2 timesteps
        concat_x = tf.concat(
            axis=0,
            values=[tf.reshape(x[current_time],
                               shape=[1, batch_size, feature_dim]),
                    tf.reshape(x[current_time + 1],
                               shape=[1, batch_size, feature_dim])])

        # Reshape to `[1, batch_size, feature_dim * 2]`
        concat_x = tf.reshape(concat_x,
                              shape=[1, batch_size, feature_dim * 2])

        tensor_list = tf.concat(axis=0, values=[tensor_list, [concat_x]])

        # Skip 2 timesteps
        current_time += 2

        return current_time, x, tensor_list
