#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Bidirectional LSTM encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class BLSTM_Encoder(object):
    """Bidirectional LSTM encoder.
    Args:
        num_units (int): the number of units in each layer
        num_layers (int): the number of layers
        num_classes (int): the number of classes of target labels.
            If 0, return hidden states before passing through the softmax layer
        lstm_impl (string, optional):
            BasicLSTMCell or LSTMCell or LSTMBlockCell or
                LSTMBlockFusedCell or CudnnLSTM.
            Choose the background implementation of tensorflow.
            Default is LSTMBlockCell (the fastest implementation).
        use_peephole (bool, optional): if True, use peephole
        parameter_init (float, optional): the range of uniform distribution to
            initialize weight parameters (>= 0)
        clip_activation (float, optional): the range of activation clipping (> 0)
        num_proj (int, optional): the number of nodes in the projection layer
        bottleneck_dim (int, optional): the dimensions of the bottleneck layer
        name (string, optional): the name of encoder
    """

    def __init__(self,
                 num_units,
                 num_layers,
                 num_classes,
                 lstm_impl='LSTMBlockCell',
                 use_peephole=True,
                 parameter_init=0.1,
                 clip_activation=5.0,
                 num_proj=None,
                 bottleneck_dim=None,
                 name='blstm_encoder'):

        self.num_units = num_units
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm_impl = lstm_impl
        self.use_peephole = use_peephole
        self.parameter_init = parameter_init
        self.clip_activation = clip_activation
        if lstm_impl != 'LSTMCell':
            self.num_proj = None
        elif num_proj not in [None, 0]:
            self.num_proj = int(num_proj)
        else:
            self.num_proj = None
        self.bottleneck_dim = int(bottleneck_dim) if bottleneck_dim not in [
            None, 0] else None
        self.name = name

        self.return_hidden_states = True if num_classes == 0 else False

    def __call__(self, inputs, inputs_seq_len,
                 keep_prob_input, keep_prob_hidden, keep_prob_output):
        """Construct model graph.
        Args:
            inputs (placeholder): A tensor of size`[B, T, input_size]`
            inputs_seq_len (placeholder): A tensor of size` [B]`
            keep_prob_input (placeholder, float): A probability to keep nodes
                in the input-hidden connection
            keep_prob_hidden (placeholder, float): A probability to keep nodes
                in the hidden-hidden connection
            keep_prob_output (placeholder, float): A probability to keep nodes
                in the hidden-output connection
        Returns:
            logits: A tensor of size `[T, B, num_classes]`
            final_state: A final hidden state of the encoder
        """
        # inputs: `[B, T, input_size]`
        batch_size = tf.shape(inputs)[0]
        input_size = tf.shape(inputs)[2]

        # Dropout for the input-hidden connection
        outputs = tf.nn.dropout(
            inputs, keep_prob_input, name='dropout_input')

        initializer = tf.random_uniform_initializer(
            minval=-self.parameter_init, maxval=self.parameter_init)

        # Hidden layers
        if self.lstm_impl == 'CudnnLSTM':
            stacked_lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
                num_layers=self.num_layers,
                num_units=self.num_units,
                input_size=input_size,
                input_mode='auto_select',
                direction='bidirectional',
                dropout=0,
                seed=0)

            params_size_t = stacked_lstm.params_size()
            init_h = tf.zeros(
                [self.num_layers * 2, batch_size, self.num_units],
                dtype=tf.float32, name="init_lstm_h")
            init_c = tf.zeros(
                [self.num_layers * 2, batch_size, self.num_units],
                dtype=tf.float32, name="init_lstm_c")
            # cudnn_params = tf.Variable(tf.random_uniform(
            #     [params_size_t], -self.parameter_init, self.parameter_init),
            #     validate_shape=False, name="lstm_params", trainable=True)
            lstm_params = tf.get_variable(
                "lstm_params",
                initializer=tf.random_uniform(
                    [params_size_t], -self.parameter_init, self.parameter_init),
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
            # raise NotImplementedError

        else:
            for i_layer in range(1, self.num_layers + 1, 1):
                with tf.variable_scope('blstm_hidden' + str(i_layer),
                                       initializer=initializer) as scope:

                    if self.lstm_impl == 'BasicLSTMCell':
                        lstm_fw = tf.contrib.rnn.BasicLSTMCell(
                            self.num_units,
                            forget_bias=1.0,
                            state_is_tuple=True,
                            activation=tf.tanh)
                        lstm_bw = tf.contrib.rnn.BasicLSTMCell(
                            self.num_units,
                            forget_bias=1.0,
                            state_is_tuple=True,
                            activation=tf.tanh)

                    elif self.lstm_impl == 'LSTMCell':
                        lstm_fw = tf.contrib.rnn.LSTMCell(
                            self.num_units,
                            use_peepholes=self.use_peephole,
                            cell_clip=self.clip_activation,
                            num_proj=self.num_proj,
                            forget_bias=1.0,
                            state_is_tuple=True)
                        lstm_bw = tf.contrib.rnn.LSTMCell(
                            self.num_units,
                            use_peepholes=self.use_peephole,
                            cell_clip=self.clip_activation,
                            num_proj=self.num_proj,
                            forget_bias=1.0,
                            state_is_tuple=True)

                    elif self.lstm_impl == 'LSTMBlockCell':
                        # NOTE: This should be faster than
                        # tf.contrib.rnn.LSTMCell
                        lstm_fw = tf.contrib.rnn.LSTMBlockCell(
                            self.num_units,
                            forget_bias=1.0,
                            # clip_cell=True,
                            use_peephole=self.use_peephole)
                        lstm_bw = tf.contrib.rnn.LSTMBlockCell(
                            self.num_units,
                            forget_bias=1.0,
                            # clip_cell=True,
                            use_peephole=self.use_peephole)
                        # TODO: cell clipping (update for rc1.3)

                    elif self.lstm_impl == 'LSTMBlockFusedCell':
                        # NOTE: This should be faster than
                        # tf.contrib.rnn.LSTMBlockFusedCell
                        lstm_fw = tf.contrib.rnn.LSTMBlockFusedCell(
                            self.num_units,
                            forget_bias=1.0,
                            # clip_cell=True,
                            use_peephole=self.use_peephole)
                        lstm_bw = tf.contrib.rnn.LSTMBlockFusedCell(
                            self.num_units,
                            forget_bias=1.0,
                            # clip_cell=True,
                            use_peephole=self.use_peephole)
                        # TODO: cell clipping (update for rc1.3)
                        lstm_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_bw)

                    else:
                        raise IndexError(
                            'lstm_impl is "BasicLSTMCell" or "LSTMCell" or ' +
                            '"LSTMBlockCell" or "LSTMBlockFusedCell" or ' +
                            '"CudnnLSTM".')

                    if self.lstm_impl == 'LSTMBlockFusedCell':
                        with tf.variable_scope("lstm_fw") as scope:
                            outputs_fw, final_state_fw = lstm_bw(
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
                    else:
                        # Dropout for the hidden-hidden connections
                        lstm_fw = tf.contrib.rnn.DropoutWrapper(
                            lstm_fw, output_keep_prob=keep_prob_hidden)
                        lstm_bw = tf.contrib.rnn.DropoutWrapper(
                            lstm_bw, output_keep_prob=keep_prob_hidden)

                        (outputs_fw, outputs_bw), final_state = tf.nn.bidirectional_dynamic_rnn(
                            cell_fw=lstm_fw,
                            cell_bw=lstm_bw,
                            inputs=outputs,
                            sequence_length=inputs_seq_len,
                            dtype=tf.float32,
                            scope=scope)
                        # NOTE: initial states are zero states by default

                        outputs = tf.concat(
                            axis=2, values=[outputs_fw, outputs_bw])

        if self.return_hidden_states:
            return outputs, final_state

        # Reshape to apply the same weights over the timesteps
        if self.num_proj is None:
            outputs = tf.reshape(outputs, shape=[-1, self.num_units * 2])
        else:
            outputs = tf.reshape(outputs, shape=[-1, self.num_proj * 2])

        if self.bottleneck_dim is not None and self.bottleneck_dim != 0:
            with tf.variable_scope('bottleneck') as scope:
                outputs = tf.contrib.layers.fully_connected(
                    outputs, self.bottleneck_dim,
                    activation_fn=tf.nn.relu,
                    weights_initializer=tf.truncated_normal_initializer(
                        stddev=self.parameter_init),
                    biases_initializer=tf.zeros_initializer(),
                    scope=scope)

                # Dropout for the hidden-output connections
                outputs = tf.nn.dropout(
                    outputs, keep_prob_output, name='dropout_output_bottle')

        with tf.variable_scope('output') as scope:
            logits_2d = tf.contrib.layers.fully_connected(
                outputs, self.num_classes,
                activation_fn=None,
                weights_initializer=tf.truncated_normal_initializer(
                    stddev=self.parameter_init),
                biases_initializer=tf.zeros_initializer(),
                scope=scope)

            # Reshape back to the original shape
            logits = tf.reshape(
                logits_2d, shape=[batch_size, -1, self.num_classes])

            # Convert to time-major: `[T, B, num_classes]'
            logits = tf.transpose(logits, (1, 0, 2))

            # Dropout for the hidden-output connections
            logits = tf.nn.dropout(
                logits, keep_prob_output, name='dropout_output')
            # NOTE: This may lead to bad results

            return logits, final_state
