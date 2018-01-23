#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Pyramidal bidirectional LSTM encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class PyramidBLSTMEncoder(object):
    """Pyramidal bidirectional LSTM Encoder.
    Args:
        num_units (int): the number of units in each layer
        num_layers (int): the number of layers
        lstm_impl (string, optional): BasicLSTMCell or LSTMCell or
            LSTMBlockCell or LSTMBlockFusedCell or CudnnLSTM.
            Choose the background implementation of tensorflow.
            Default is LSTMBlockCell.
        use_peephole (bool, optional): if True, use peephole
        parameter_init (float, optional): the range of uniform distribution to
            initialize weight parameters (>= 0)
        clip_activation (float, optional): the range of activation clipping (> 0)
        # num_proj (int, optional): the number of nodes in the projection layer
        concat (bool, optional):
        name (string, optional): the name of encoder
    """

    def __init__(self,
                 num_units,
                 num_layers,
                 lstm_impl,
                 use_peephole,
                 parameter_init,
                 clip_activation,
                 num_proj,
                 concat=False,
                 name='pblstm_encoder'):

        assert num_proj != 0
        assert num_units % 2 == 0, 'num_unit should be even number.'

        self.num_units = num_units
        self.num_proj = None
        self.num_layers = num_layers
        self.lstm_impl = lstm_impl
        self.use_peephole = use_peephole
        self.parameter_init = parameter_init
        self.clip_activation = clip_activation
        self.name = name

    def _build(self, inputs, inputs_seq_len, keep_prob, is_training):
        """Construct Pyramidal Bidirectional LSTM encoder.
        Args:
            inputs (placeholder): A tensor of size`[B, T, input_size]`
            inputs_seq_len (placeholder): A tensor of size` [B]`
            keep_prob (placeholder, float): A probability to keep nodes
                in the hidden-hidden connection
            is_training (bool):
        Returns:
            outputs: Encoder states, a tensor of size
                `[T, B, num_units (num_proj)]`
            final_state: A final hidden state of the encoder
        """
        initializer = tf.random_uniform_initializer(
            minval=-self.parameter_init, maxval=self.parameter_init)

        # Hidden layers
        outputs = inputs
        for i_layer in range(1, self.num_layers + 1, 1):
            with tf.variable_scope('pblstm_hidden' + str(i_layer),
                                   initializer=initializer) as scope:

                lstm_fw = tf.contrib.rnn.LSTMCell(
                    self.num_units,
                    use_peepholes=self.use_peephole,
                    cell_clip=self.clip_activation,
                    initializer=initializer,
                    num_proj=None,
                    forget_bias=1.0,
                    state_is_tuple=True)
                lstm_bw = tf.contrib.rnn.LSTMCell(
                    self.num_units,
                    use_peepholes=self.use_peephole,
                    cell_clip=self.clip_activation,
                    initializer=initializer,
                    num_proj=self.num_proj,
                    forget_bias=1.0,
                    state_is_tuple=True)

                # Dropout for the hidden-hidden connections
                lstm_fw = tf.contrib.rnn.DropoutWrapper(
                    lstm_fw, output_keep_prob=keep_prob)
                lstm_bw = tf.contrib.rnn.DropoutWrapper(
                    lstm_bw, output_keep_prob=keep_prob)

                if i_layer > 0:
                    # Convert to time-major: `[T, B, input_size]`
                    outputs = tf.transpose(outputs, (1, 0, 2))
                    max_time = tf.shape(outputs)[0]

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

                (outputs_fw, outputs_bw), final_state = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=lstm_fw,
                    cell_bw=lstm_bw,
                    inputs=outputs,
                    sequence_length=inputs_seq_len,
                    dtype=tf.float32,
                    time_major=False,
                    scope=scope)
                # NOTE: initial states are zero states by default

                # Concatenate each direction
                outputs = tf.concat(axis=2, values=[outputs_fw, outputs_bw])

        return outputs, final_state

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
