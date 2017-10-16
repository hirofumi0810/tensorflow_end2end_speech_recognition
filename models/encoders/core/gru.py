#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""GRU encoders."""

import tensorflow as tf


class GRUEncoder(object):
    """Unidirectional GRU encoder.
    Args:
        num_units (int): the number of units in each layer
        num_layers (int): the number of layers
        parameter_init (float, optional): the range of uniform distribution to
            initialize weight parameters (>= 0)
        time_major (bool, optional): if True, time-major computation will be
            performed
        name (string, optional): the name of encoder
    """

    def __init__(self,
                 num_units,
                 num_layers,
                 parameter_init,
                 time_major=False,
                 name='gru_encoder'):

        self.num_units = num_units
        self.num_layers = num_layers
        self.parameter_init = parameter_init
        self.time_major = time_major
        self.name = name

    def __call__(self, inputs, inputs_seq_len, keep_prob):
        """Construct model graph.
        Args:
            inputs (placeholder): A tensor of size`[B, T, input_size]`
            inputs_seq_len (placeholder): A tensor of size` [B]`
            keep_prob (placeholder, float): A probability to keep nodes
                in the hidden-hidden connection
        Returns:
            outputs: Encoder states, a tensor of size `[T, B, num_units]`
            final_state: A final hidden state of the encoder
        """
        if self.time_major:
            # Convert form batch-major to time-major
            inputs = tf.transpose(inputs, [1, 0, 2])

        initializer = tf.random_uniform_initializer(
            minval=-self.parameter_init, maxval=self.parameter_init)

        # Hidden layers
        gru_list = []
        with tf.variable_scope('multi_gru', initializer=initializer) as scope:
            for i_layer in range(1, self.num_layers + 1, 1):

                gru = tf.contrib.rnn.GRUCell(self.num_units)

                # Dropout for the hidden-hidden connections
                gru = tf.contrib.rnn.DropoutWrapper(
                    gru, output_keep_prob=keep_prob)

                gru_list.append(gru)

            # Stack multiple cells
            stacked_gru = tf.contrib.rnn.MultiRNNCell(
                gru_list, state_is_tuple=True)

            # Ignore 2nd return (the last state)
            outputs, final_state = tf.nn.dynamic_rnn(
                cell=stacked_gru,
                inputs=inputs,
                sequence_length=inputs_seq_len,
                dtype=tf.float32,
                time_major=self.time_major)
            # NOTE: initial states are zero states by default

        return outputs, final_state


class BGRUEncoder(object):
    """Bidirectional GRU encoder.
    Args:
        num_units (int): the number of units in each layer
        num_layers (int): the number of layers
        parameter_init (float, optional): the range of uniform distribution to
            initialize weight parameters (>= 0)
        time_major (bool, optional): if True, time-major computation will be
            performed
        name (string, optional): the name of the encoder
    """

    def __init__(self,
                 num_units,
                 num_layers,
                 parameter_init,
                 time_major=False,
                 name='bgru_encoder'):

        self.num_units = num_units
        self.num_layers = num_layers
        self.parameter_init = parameter_init
        self.time_major = time_major
        self.name = name

    def __call__(self, inputs, inputs_seq_len, keep_prob):
        """Construct model graph.
        Args:
            inputs (placeholder): A tensor of size`[B, T, input_size]`
            inputs_seq_len (placeholder): A tensor of size` [B]`
            keep_prob (placeholder, float): A probability to keep nodes
                in the hidden-hidden connection
        Returns:
            outputs: Encoder states, a tensor of size　`[T, B, num_units]`
            final_state: A final hidden state of the encoder
        """
        if self.time_major:
            # Convert form batch-major to time-major
            inputs = tf.transpose(inputs, [1, 0, 2])

        initializer = tf.random_uniform_initializer(
            minval=-self.parameter_init, maxval=self.parameter_init)

        # Hidden layers
        outputs = inputs
        for i_layer in range(1, self.num_layers + 1, 1):
            with tf.variable_scope('bgru_hidden' + str(i_layer),
                                   initializer=initializer) as scope:

                gru_fw = tf.contrib.rnn.GRUCell(self.num_units)
                gru_bw = tf.contrib.rnn.GRUCell(self.num_units)

                # Dropout for the hidden-hidden connections
                gru_fw = tf.contrib.rnn.DropoutWrapper(
                    gru_fw, output_keep_prob=keep_prob)
                gru_bw = tf.contrib.rnn.DropoutWrapper(
                    gru_bw, output_keep_prob=keep_prob)

                # Ignore 2nd return (the last state)
                (outputs_fw, outputs_bw), final_state = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=gru_fw,
                    cell_bw=gru_bw,
                    inputs=outputs,
                    sequence_length=inputs_seq_len,
                    dtype=tf.float32,
                    time_major=self.time_major,
                    scope=scope)
                # NOTE: initial states are zero states by default

                outputs = tf.concat(axis=2, values=[outputs_fw, outputs_bw])

        return outputs, final_state
