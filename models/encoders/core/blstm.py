#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Bidirectional LSTM encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class BLSTMEncoder(object):
    """Bidirectional LSTM encoder.
    Args:
        num_units (int): the number of units in each layer
        num_proj (int): the number of nodes in the projection layer
        num_layers (int): the number of layers
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
                 num_layers,
                 lstm_impl,
                 use_peephole,
                 parameter_init,
                 clip_activation,
                 time_major=True,
                 name='lstm_encoder'):
        if num_proj == 0:
            raise ValueError

        self.num_units = num_units
        if lstm_impl != 'LSTMCell':
            self.num_proj = None
        else:
            self.num_proj = num_proj
        # TODO: fix this
        self.num_layers = num_layers
        self.lstm_impl = lstm_impl
        self.use_peephole = use_peephole
        self.parameter_init = parameter_init
        self.clip_activation = clip_activation
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
            outputs: Encoder states.
                if time_major is True, a tensor of size
                    `[T, B, num_units (num_proj)]`
                otherwise, `[B, T, num_units (num_proj)]`
            final_state: A final hidden state of the encoder
        """
        initializer = tf.random_uniform_initializer(
            minval=-self.parameter_init, maxval=self.parameter_init)

        if self.lstm_impl == 'BasicLSTMCell':
            outputs, final_state = basiclstmcell(
                self.num_units, self.num_layers,
                inputs, inputs_seq_len, keep_prob, initializer,
                self.time_major)

        elif self.lstm_impl == 'LSTMCell':
            outputs, final_state = lstmcell(
                self.num_units, self.num_proj, self.num_layers,
                self.use_peephole, self.clip_activation,
                inputs, inputs_seq_len, keep_prob, initializer,
                self.time_major)

        elif self.lstm_impl == 'LSTMBlockCell':
            outputs, final_state = lstmblockcell(
                self.num_units, self.num_layers,
                self.use_peephole, self.clip_activation,
                inputs, inputs_seq_len, keep_prob, initializer,
                self.time_major)

        elif self.lstm_impl == 'LSTMBlockFusedCell':
            outputs, final_state = lstmblockfusedcell(
                self.num_units, self.num_layers,
                self.use_peephole, self.clip_activation,
                inputs, inputs_seq_len, keep_prob, initializer,
                self.time_major)

        elif self.lstm_impl == 'CudnnLSTM':
            outputs, final_state = cudnnlstm(
                self.num_units, self.num_layers, self.parameter_init,
                inputs, inputs_seq_len, keep_prob, initializer,
                self.time_major)

        else:
            raise IndexError(
                'lstm_impl is "BasicLSTMCell" or "LSTMCell" or ' +
                '"LSTMBlockCell" or "LSTMBlockFusedCell" or ' +
                '"CudnnLSTM".')

        return outputs, final_state


def basiclstmcell(num_units, num_layers, inputs, inputs_seq_len, keep_prob,
                  initializer, time_major, num_layers_sub=None):

    if time_major:
        # Convert form batch-major to time-major
        inputs = tf.transpose(inputs, [1, 0, 2])

    outputs = inputs
    for i_layer in range(1, num_layers + 1, 1):
        with tf.variable_scope('blstm_hidden' + str(i_layer),
                               initializer=initializer) as scope:

            lstm_fw = tf.contrib.rnn.BasicLSTMCell(
                num_units,
                forget_bias=1.0,
                state_is_tuple=True,
                activation=tf.tanh)
            lstm_bw = tf.contrib.rnn.BasicLSTMCell(
                num_units,
                forget_bias=1.0,
                state_is_tuple=True,
                activation=tf.tanh)

            # Dropout for the hidden-hidden connections
            lstm_fw = tf.contrib.rnn.DropoutWrapper(
                lstm_fw, output_keep_prob=keep_prob)
            lstm_bw = tf.contrib.rnn.DropoutWrapper(
                lstm_bw, output_keep_prob=keep_prob)

            (outputs_fw, outputs_bw), final_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=lstm_fw,
                cell_bw=lstm_bw,
                inputs=outputs,
                sequence_length=inputs_seq_len,
                dtype=tf.float32,
                time_major=time_major,
                scope=scope)
            # NOTE: initial states are zero states by default

            outputs = tf.concat(axis=2, values=[outputs_fw, outputs_bw])

            if num_layers_sub is not None and i_layer == num_layers_sub:
                outputs_sub = outputs
                final_state_sub = final_state

    if num_layers_sub is not None:
        return outputs, final_state, outputs_sub, final_state_sub
    else:
        return outputs, final_state


def lstmcell(num_units, num_proj, num_layers, use_peephole, clip_activation,
             inputs, inputs_seq_len, keep_prob, initializer, time_major,
             num_layers_sub=None):

    if time_major:
        # Convert form batch-major to time-major
        inputs = tf.transpose(inputs, [1, 0, 2])

    outputs = inputs
    for i_layer in range(1, num_layers + 1, 1):
        with tf.variable_scope('blstm_hidden' + str(i_layer),
                               initializer=initializer) as scope:

            lstm_fw = tf.contrib.rnn.LSTMCell(
                num_units,
                use_peepholes=use_peephole,
                cell_clip=clip_activation,
                num_proj=num_proj,
                forget_bias=1.0,
                state_is_tuple=True)
            lstm_bw = tf.contrib.rnn.LSTMCell(
                num_units,
                use_peepholes=use_peephole,
                cell_clip=clip_activation,
                num_proj=num_proj,
                forget_bias=1.0,
                state_is_tuple=True)

            # Dropout for the hidden-hidden connections
            lstm_fw = tf.contrib.rnn.DropoutWrapper(
                lstm_fw, output_keep_prob=keep_prob)
            lstm_bw = tf.contrib.rnn.DropoutWrapper(
                lstm_bw, output_keep_prob=keep_prob)

            (outputs_fw, outputs_bw), final_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=lstm_fw,
                cell_bw=lstm_bw,
                inputs=outputs,
                sequence_length=inputs_seq_len,
                dtype=tf.float32,
                time_major=time_major,
                scope=scope)
            # NOTE: initial states are zero states by default

            outputs = tf.concat(axis=2, values=[outputs_fw, outputs_bw])

            if num_layers_sub is not None and i_layer == num_layers_sub:
                outputs_sub = outputs
                final_state_sub = final_state

    if num_layers_sub is not None:
        return outputs, final_state, outputs_sub, final_state_sub
    else:
        return outputs, final_state


def lstmblockcell(num_units, num_layers, use_peephole, clip_activation,
                  inputs, inputs_seq_len, keep_prob, initializer, time_major,
                  num_layers_sub=None):

    if time_major:
        # Convert form batch-major to time-major
        inputs = tf.transpose(inputs, [1, 0, 2])

    outputs = inputs
    for i_layer in range(1, num_layers + 1, 1):
        with tf.variable_scope('blstm_hidden' + str(i_layer),
                               initializer=initializer) as scope:

            if tf.__version__ == '1.3.0':
                lstm_fw = tf.contrib.rnn.LSTMBlockCell(
                    num_units,
                    forget_bias=1.0,
                    clip_cell=clip_activation,
                    use_peephole=use_peephole)
                lstm_bw = tf.contrib.rnn.LSTMBlockCell(
                    num_units,
                    forget_bias=1.0,
                    clip_cell=clip_activation,
                    use_peephole=use_peephole)
            else:
                lstm_fw = tf.contrib.rnn.LSTMBlockCell(
                    num_units,
                    forget_bias=1.0,
                    use_peephole=use_peephole)
                lstm_bw = tf.contrib.rnn.LSTMBlockCell(
                    num_units,
                    forget_bias=1.0,
                    use_peephole=use_peephole)

            # Dropout for the hidden-hidden connections
            lstm_fw = tf.contrib.rnn.DropoutWrapper(
                lstm_fw, output_keep_prob=keep_prob)
            lstm_bw = tf.contrib.rnn.DropoutWrapper(
                lstm_bw, output_keep_prob=keep_prob)

            (outputs_fw, outputs_bw), final_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=lstm_fw,
                cell_bw=lstm_bw,
                inputs=outputs,
                sequence_length=inputs_seq_len,
                dtype=tf.float32,
                time_major=time_major,
                scope=scope)
            # NOTE: initial states are zero states by default

            outputs = tf.concat(axis=2, values=[outputs_fw, outputs_bw])

            if num_layers_sub is not None and i_layer == num_layers_sub:
                outputs_sub = outputs
                final_state_sub = final_state

    if num_layers_sub is not None:
        return outputs, final_state, outputs_sub, final_state_sub
    else:
        return outputs, final_state


def lstmblockfusedcell(num_units, num_layers, use_peephole, clip_activation,
                       inputs, inputs_seq_len, keep_prob, initializer,
                       time_major, num_layers_sub=None):

    # TODO: add time-major

    outputs = inputs
    for i_layer in range(1, num_layers + 1, 1):
        with tf.variable_scope('blstm_hidden' + str(i_layer),
                               initializer=initializer) as scope:

            if tf.__version__ == '1.3.0':
                lstm_fw = tf.contrib.rnn.LSTMBlockFusedCell(
                    num_units,
                    forget_bias=1.0,
                    clip_cell=clip_activation,
                    use_peephole=use_peephole)
                lstm_bw = tf.contrib.rnn.LSTMBlockFusedCell(
                    num_units,
                    forget_bias=1.0,
                    clip_cell=clip_activation,
                    use_peephole=use_peephole)
            else:
                lstm_fw = tf.contrib.rnn.LSTMBlockFusedCell(
                    num_units,
                    forget_bias=1.0,
                    use_peephole=use_peephole)
                lstm_bw = tf.contrib.rnn.LSTMBlockFusedCell(
                    num_units,
                    forget_bias=1.0,
                    use_peephole=use_peephole)

            lstm_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_bw)

            with tf.variable_scope("lstm_fw") as scope:
                outputs_fw, final_state_fw = lstm_fw(
                    outputs, dtype=tf.float32,
                    sequence_length=inputs_seq_len, scope=scope)
            with tf.variable_scope("lstm_bw") as scope:
                outputs_bw, final_state_bw = lstm_bw(
                    outputs, dtype=tf.float32,
                    sequence_length=inputs_seq_len, scope=scope)
            final_state = tf.contrib.rnn.LSTMStateTuple(
                h=final_state_fw, c=final_state_bw)

            # TODO: add dropout

            # outputs = tf.concat_v2([outputs_fw, outputs_bw], 2, name="output")
            outputs = tf.concat(
                axis=2, values=[outputs_fw, outputs_bw])

            # if self.num_proj > 0:
            #     outputs = tf.contrib.layers.fully_connected(
            #         activation_fn=None, inputs=outputs,
            #         num_outputs=self.num_proj, scope="projection")
            # TODO: add projection layers

    return outputs, final_state


def cudnnlstm(num_units, num_layers, parameter_init,
              inputs, inputs_seq_len, keep_prob, initializer, time_major,
              num_layers_sub=None):

    batch_size = tf.shape(inputs)[0]
    input_size = tf.shape(inputs)[2]

    # TODO: add time-major

    stacked_lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
        num_layers=num_layers,
        num_units=num_units,
        input_size=input_size,
        input_mode='auto_select',
        direction='bidirectional',
        dropout=0,
        seed=0)

    params_size_t = stacked_lstm.params_size()
    init_h = tf.zeros(
        [num_layers * 2, batch_size, num_units],
        dtype=tf.float32, name="init_lstm_h")
    init_c = tf.zeros(
        [num_layers * 2, batch_size, num_units],
        dtype=tf.float32, name="init_lstm_c")
    # cudnn_params = tf.Variable(tf.random_uniform(
    #     [params_size_t], -self.parameter_init, self.parameter_init),
    #     validate_shape=False, name="lstm_params", trainable=True)
    lstm_params = tf.get_variable(
        "lstm_params",
        initializer=tf.random_uniform(
            [params_size_t], -parameter_init, parameter_init),
        validate_shape=False,
        trainable=True)
    # TODO is_training=is_training should be changed!

    # outputs = tf.contrib.layers.fully_connected(
    #     activation_fn=None, inputs=outputs,
    #     num_outputs=nproj, scope="projection")
    # TODO: add projection layers

    outputs, output_h, output_c = stacked_lstm(
        input_data=inputs,
        input_h=init_h,
        input_c=init_c,
        params=lstm_params,
        is_training=True)
    # NOTE: outputs: `[T, B, num_units * num_direction]`

    final_state = tf.contrib.rnn.LSTMStateTuple(h=output_h, c=output_c)
    # TODO: add dropout

    return outputs, final_state
