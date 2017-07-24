#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""GRU-CTC model."""

import tensorflow as tf
from models.ctc.core.ctc_base import ctcBase


class GRU_CTC(ctcBase):
    """GRU-CTC model.
    Args:
        batch_size: int, batch size of mini batch
        input_size: int, the dimensions of input vectors
        num_unit: int, the number of units in each layer
        num_layer: int, the number of layers
        num_classes: int, the number of classes of target labels
            (except for a blank label)
        parameter_init: A float value. Range of uniform distribution to
            initialize weight parameters
        clip_grad: A float value. Range of gradient clipping (> 0)
        clip_activation: A float value. Range of activation clipping (> 0)
        dropout_ratio_input: A float value. Dropout ratio in the input-hidden
            layer
        dropout_ratio_hidden: A float value. Dropout ratio in the hidden-hidden
            layers
        dropout_ratio_output: A float value. Dropout ratio in the hidden-output
            layer
        num_proj: not used
        weight_decay: A float value. Regularization parameter for weight decay
        bottleneck_dim: int, the dimensions of the bottleneck layer
    """

    def __init__(self,
                 batch_size,
                 input_size,
                 num_unit,
                 num_layer,
                 num_classes,
                 parameter_init=0.1,
                 clip_grad=None,
                 clip_activation=None,
                 dropout_ratio_input=1.0,
                 dropout_ratio_hidden=1.0,
                 dropout_ratio_output=1.0,
                 num_proj=None,  # not used
                 weight_decay=0.0,
                 bottleneck_dim=None,
                 name='gru_ctc'):

        ctcBase.__init__(self, batch_size, input_size, num_unit, num_layer,
                         num_classes, parameter_init,
                         clip_grad, clip_activation,
                         dropout_ratio_input, dropout_ratio_hidden,
                         dropout_ratio_output, weight_decay, name)

        self.bottleneck_dim = bottleneck_dim

    def _build(self, inputs, inputs_seq_len, keep_prob_input,
               keep_prob_hidden, keep_prob_output):
        """Construct model graph.
        Args:
            inputs: A tensor of `[batch_size, max_time, input_dim]`
            inputs_seq_len:  A tensor of `[batch_size]`
            keep_prob_input: A float value. A probability to keep nodes in
                the input-hidden layer
            keep_prob_hidden: A float value. A probability to keep nodes in
                the hidden-hidden layers
            keep_prob_output: A float value. A probability to keep nodes in
                the hidden-output layer
        Returns:
            logits:
        """
        # Dropout for the input-hidden connection
        inputs = tf.nn.dropout(inputs,
                               keep_prob_input,
                               name='dropout_input')

        # Hidden layers
        gru_list = []
        for i_layer in range(self.num_layer):
            with tf.name_scope('gru_hidden' + str(i_layer + 1)):

                initializer = tf.random_uniform_initializer(
                    minval=-self.parameter_init,
                    maxval=self.parameter_init)

                with tf.variable_scope('gru', initializer=initializer):
                    gru = tf.contrib.rnn.GRUCell(self.num_unit)

                # Dropout for the hidden-hidden connections
                gru = tf.contrib.rnn.DropoutWrapper(
                    gru, output_keep_prob=keep_prob_hidden)

                gru_list.append(gru)

        # Stack multiple cells
        stacked_gru = tf.contrib.rnn.MultiRNNCell(
            gru_list, state_is_tuple=True)

        # Ignore 2nd return (the last state)
        outputs, final_state = tf.nn.dynamic_rnn(cell=stacked_gru,
                                                 inputs=inputs,
                                                 sequence_length=inputs_seq_len,
                                                 dtype=tf.float32)

        # inputs: `[batch_size, max_time, input_size]`
        batch_size = tf.shape(inputs)[0]

        # Reshape to apply the same weights over the timesteps
        output_node = self.num_unit
        outputs = tf.reshape(outputs, shape=[-1, output_node])

        if self.bottleneck_dim is not None and self.bottleneck_dim != 0:
            with tf.name_scope('bottleneck'):
                # Affine
                W_bottleneck = tf.Variable(tf.truncated_normal(
                    shape=[output_node, self.bottleneck_dim],
                    stddev=0.1, name='W_bottleneck'))
                b_bottleneck = tf.Variable(tf.zeros(
                    shape=[self.bottleneck_dim], name='b_bottleneck'))
                outputs = tf.matmul(outputs, W_bottleneck) + b_bottleneck
                output_node = self.bottleneck_dim

                # Dropout for the hidden-output connections
                outputs = tf.nn.dropout(outputs,
                                        keep_prob_output,
                                        name='dropout_output_bottle')

        with tf.name_scope('output'):
            # Affine
            W_output = tf.Variable(tf.truncated_normal(
                shape=[output_node, self.num_classes],
                stddev=0.1, name='W_output'))
            b_output = tf.Variable(tf.zeros(
                shape=[self.num_classes], name='b_output'))
            logits_2d = tf.matmul(outputs, W_output) + b_output

            # Reshape back to the original shape
            logits = tf.reshape(
                logits_2d, shape=[batch_size, -1, self.num_classes])

            # Convert to time-major: `[max_time, batch_size, num_classes]'
            logits = tf.transpose(logits, (1, 0, 2))

            # Dropout for the hidden-output connections
            logits = tf.nn.dropout(logits,
                                   keep_prob_output,
                                   name='dropout_output')

            return logits
