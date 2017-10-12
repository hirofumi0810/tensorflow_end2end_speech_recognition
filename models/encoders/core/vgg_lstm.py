#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""VGG + unidirectional LSTM encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from models.encoders.core.cnn_util import conv_layer, max_pool


class VGG_LSTM_Encoder(object):
    """VGG + unidirectional LSTM encoder.
    Args:
        input_size (int): the dimensions of input vectors
        splice (int): frames to splice
        num_units (int): the number of units in each layer
        num_layers (int): the number of layers
        num_classes (int): the number of classes of target labels.
            If 0, return hidden states before passing through the softmax layer
        lstm_impl (string, optional):
            BasicLSTMCell or LSTMCell or LSTMBlockCell or LSTMBlockFusedCell.
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
                 input_size,
                 splice,
                 num_units,
                 num_layers,
                 num_classes,
                 lstm_impl='LSTMBlockCell',
                 use_peephole=True,
                 parameter_init=0.1,
                 clip_activation=5.0,
                 num_proj=None,
                 bottleneck_dim=None,
                 name='vgg_lstm_encoder'):

        self.input_size = input_size
        self.splice = splice
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
        # inputs: `[B, T, input_size * splice]`
        batch_size = tf.shape(inputs)[0]
        max_time = tf.shape(inputs)[1]

        # Reshape to 4D tensor `[B * T, input_size / 3, splice, 3(+Δ, ΔΔ)]`
        inputs = tf.reshape(
            inputs,
            shape=[batch_size * max_time, int(self.input_size / 3), self.splice, 3])

        with tf.variable_scope('VGG1'):
            inputs = conv_layer(inputs,
                                filter_shape=[3, 3, 3, 64],
                                parameter_init=self.parameter_init,
                                relu=True,
                                name='conv1')
            inputs = conv_layer(inputs,
                                filter_shape=[3, 3, 64, 64],
                                parameter_init=self.parameter_init,
                                relu=True,
                                name='conv2')
            inputs = max_pool(inputs, name='max_pool')
            # TODO(hirofumi): try batch normalization

        with tf.variable_scope('VGG2'):
            inputs = conv_layer(inputs,
                                filter_shape=[3, 3, 64, 128],
                                parameter_init=self.parameter_init,
                                relu=True,
                                name='conv1')
            inputs = conv_layer(inputs,
                                filter_shape=[3, 3, 128, 128],
                                parameter_init=self.parameter_init,
                                relu=True,
                                name='conv2')
            inputs = max_pool(inputs, name='max_pool')
            # TODO(hirofumi): try batch normalization

        # Reshape to 2D tensor `[B * T, new_h * new_w * 128]`
        new_h = math.ceil(self.input_size / 3 / 4)  # expected to be 11 ro 10
        new_w = math.ceil(self.splice / 4)  # expected to be 3
        inputs = tf.reshape(
            inputs, shape=[batch_size * max_time, new_h * new_w * 128])

        # Insert linear layer to recude CNN's output demention
        # from (new_h * new_w * 128) to 256
        with tf.variable_scope('linear') as scope:
            inputs = tf.contrib.layers.fully_connected(
                inputs=inputs,
                num_outputs=256,
                activation_fn=tf.nn.relu,
                scope=scope)

        # Dropout for the VGG-output-hidden connection
        inputs = tf.nn.dropout(inputs,
                               keep_prob_input,
                               name='dropout_input')

        # Reshape back to 3D tensor `[B, T, 256]`
        inputs = tf.reshape(inputs, shape=[batch_size, max_time, 256])

        initializer = tf.random_uniform_initializer(
            minval=-self.parameter_init,
            maxval=self.parameter_init)

        # Hidden layers
        lstm_list = []
        with tf.variable_scope('multi_lstm', initializer=initializer) as scope:
            for i_layer in range(1, self.num_layers + 1, 1):

                if self.lstm_impl == 'BasicLSTMCell':
                    lstm = tf.contrib.rnn.BasicLSTMCell(
                        self.num_units,
                        forget_bias=1.0,
                        state_is_tuple=True,
                        activation=tf.tanh)

                elif self.lstm_impl == 'LSTMCell':
                    lstm = tf.contrib.rnn.LSTMCell(
                        self.num_units,
                        use_peepholes=self.use_peephole,
                        cell_clip=self.clip_activation,
                        num_proj=self.num_proj,
                        forget_bias=1.0,
                        state_is_tuple=True)

                elif self.lstm_impl == 'LSTMBlockCell':
                    # NOTE: This should be faster than tf.contrib.rnn.LSTMCell
                    lstm = tf.contrib.rnn.LSTMBlockCell(
                        self.num_units,
                        forget_bias=1.0,
                        # clip_cell=True,
                        use_peephole=self.use_peephole)
                    # TODO: cell clipping (update for rc1.3)

                elif self.lstm_impl == 'LSTMBlockFusedCell':
                    raise NotImplementedError

                elif self.lstm_impl == 'CudnnLSTM':
                    raise NotImplementedError

                else:
                    raise IndexError(
                        'lstm_impl is "BasicLSTMCell" or "LSTMCell" or ' +
                        '"LSTMBlockCell" or "LSTMBlockFusedCell" or ' +
                        '"CudnnLSTM".')

                # Dropout for the hidden-hidden connections
                lstm = tf.contrib.rnn.DropoutWrapper(
                    lstm, output_keep_prob=keep_prob_hidden)

                lstm_list.append(lstm)

            # Stack multiple cells
            stacked_lstm = tf.contrib.rnn.MultiRNNCell(
                lstm_list, state_is_tuple=True)

            # Ignore 2nd return (the last state)
            outputs, final_state = tf.nn.dynamic_rnn(
                cell=stacked_lstm,
                inputs=inputs,
                sequence_length=inputs_seq_len,
                dtype=tf.float32,
                scope=scope)
            # NOTE: initial states are zero states by default

        if self.return_hidden_states:
            return outputs, final_state

        # Reshape to apply the same weights over the timesteps
        if self.num_proj is None:
            outputs = tf.reshape(outputs, shape=[-1, self.num_units])
        else:
            outputs = tf.reshape(outputs, shape=[-1, self.num_proj])

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

            return logits, final_state
