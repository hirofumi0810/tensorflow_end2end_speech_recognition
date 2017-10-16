#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Attention layer for computing attention weights."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

ATTENTION_TYPE = ['content', 'location', 'hybrid', 'MLP_dot',
                  'luong_dot', 'luong_general', 'luong_concat',
                  'baidu_attetion']


class AttentionLayer(object):
    """Attention layer.
    Args:
        attention_type: string, content or location or hybrid or layer_dot
        num_units (int): the number of units used in the attention layer
        attention_weights_tempareture (float): a smoothing factor in the
            softmax layer for computing attention weights
        name (string): the name of the attention layer
    """

    def __init__(self, attention_type, num_units,
                 attention_weights_tempareture, name='attention_layer'):
        self.attention_type = attention_type
        self.num_units = num_units
        self.attention_weights_tempareture = attention_weights_tempareture
        self.name = name

    def __call__(self, encoder_outputs, decoder_output,
                 encoder_outputs_length, attention_weights):
        """Computes attention scores and outputs.
        Args:
            encoder_outputs: The outputs of the encoder. A tensor of size
                `[B, T_in, encoder_num_units]`
            decoder_output: The current state of the docoder. A tensor of size
                `[B, decoder_num_units]`
            encoder_outputs_length: An int32 tensor of size `[B]` defining
                the sequence length of `encoder_outputs`
            attention_weights: This is used for the location-based
                attention. A tensor of size `[B, T_in]`
        Returns:
            A tuple `(attention_weights, context_vector)`.
                attention_weights: vector of length `T_in` where each
                    element is the normalized "score" of the corresponding
                    `inputs` element. A tensor of size `[B, T_in]`
                context_vector: the final attention layer output
                    corresponding to the weighted inputs.
                    A tensor of size `[B, encoder_num_units]`.
        """
        # Compute attention scores over encoder outputs
        energy = self._compute_attention_score(
            attention_type=self.attention_type,
            encoder_outputs=encoder_outputs,
            decoder_output=decoder_output,
            attention_weights=attention_weights)

        # Replace all scores for padded inputs with tf.float32.min
        max_time_input = tf.shape(energy)[1]
        scores_mask = tf.sequence_mask(
            lengths=tf.to_int32(encoder_outputs_length),
            maxlen=tf.to_int32(max_time_input),
            dtype=tf.float32)
        # ex.) tf.sequence_mask
        # tf.sequence_mask([1, 3, 2], 5) = [[True, False, False, False, False],
        #                                   [True, True, True, False, False],
        #                                   [True, True, False, False, False]]
        energy = energy * scores_mask + ((1.0 - scores_mask) * tf.float32.min)
        # NOTE: For underflow

        # Smoothing
        energy /= self.attention_weights_tempareture

        # Compute attention weights
        attention_weights = tf.nn.softmax(
            energy, name="attention_weights")
        # attention_weights = tf.exp(energy) / tf.reduce_sum(
        #     tf.exp(energy),
        #     axis=-1,
        #     keep_dims=True)

        # Compute context vector
        context_vector = tf.expand_dims(
            attention_weights, axis=2) * encoder_outputs
        context_vector = tf.reduce_sum(
            context_vector, axis=1, name="context_vector")
        encoder_num_units = encoder_outputs.get_shape().as_list()[-1]
        context_vector.set_shape([None, encoder_num_units])

        return attention_weights, context_vector

    def _compute_attention_score(self, attention_type, encoder_outputs,
                                 decoder_output, attention_weights):
        """An attention layer that calculates attention scores.
        Args:
            attention_type (string): the type of attention
            encoder_outputs (): The sequence of encoder outputs
                A tensor of shape `[B, T_in, encoder_num_units]`
            decoder_output: The current state of the docoder
                A tensor of shape `[B, decoder_num_units]`
            attention_weights: A tensor of size `[B, T_in]`
        Returns:
            attention_sum: The summation of attention scores
                A tensor of shape `[B, T_in]`
        """
        if attention_type not in ATTENTION_TYPE:
            raise ValueError(
                "attention type should be one of [%s], you provided %s." %
                (", ".join(ATTENTION_TYPE), attention_type))

        # Fully connected layers to transform both encoder_outputs and
        # decoder_output into a tensor with `num_units` units
        W_dec = tf.contrib.layers.fully_connected(
            decoder_output,
            num_outputs=self.num_units,
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(
                stddev=self.parameter_init),
            biases_initializer=tf.zeros_initializer(),
            scope="W_decoder")
        # NOTE: `[B, num_units]`

        if attention_type == 'MLP_dot':
            ############################################################
            # layer-dot attention
            # energy = dot(W_enc(hidden_enc), W_dec(hidden_dec))
            ############################################################
            W_enc = tf.contrib.layers.fully_connected(
                encoder_outputs,
                num_outputs=self.num_units,
                activation_fn=None,
                weights_initializer=tf.truncated_normal_initializer(
                    stddev=self.parameter_init),
                biases_initializer=tf.zeros_initializer(),
                scope="W_encoder")

            # Calculates a batch- and time-wise dot product
            energy = W_enc * tf.expand_dims(W_dec, axis=1)

        elif attention_type == 'content':
            ############################################################
            # content-based attention
            # energy = dot(v_a, tanh(W_enc(hidden_enc) + W_dec(hidden_dec)))
            ############################################################
            W_enc = tf.contrib.layers.fully_connected(
                encoder_outputs,
                num_outputs=self.num_units,
                activation_fn=None,
                weights_initializer=tf.truncated_normal_initializer(
                    stddev=self.parameter_init),
                biases_initializer=tf.zeros_initializer(),
                scope="W_encoder")

            v_a = tf.get_variable(
                "v_a", shape=[self.num_units], dtype=tf.float32)

            # Calculates a batch- and time-wise dot product with a variable
            energy = v_a * tf.tanh(W_enc + tf.expand_dims(W_dec, axis=1))

        elif attention_type == 'hybrid':
            ############################################################
            # hybrid attention (content-based + location-based)
            # f = F * α_{i-1}
            # energy = dot(v_a, tanh(W_enc(hidden_enc) + W_dec(hidden_dec) + W_fil(f)))
            ############################################################
            W_enc = tf.contrib.layers.fully_connected(
                encoder_outputs,
                num_outputs=self.num_units,
                activation_fn=None,
                weights_initializer=tf.truncated_normal_initializer(
                    stddev=self.parameter_init),
                biases_initializer=tf.zeros_initializer(),
                scope="W_encoder")

            with tf.control_dependencies(None):
                F = tf.Variable(tf.truncated_normal(
                    shape=[200, 1, 10],
                    # shape=[100, 1, 10],
                    stddev=0.1), name='filter')
            # `[B, T_in]` -> `[B, T_in, 1]`
            attention_weights = tf.expand_dims(attention_weights, axis=2)
            f = tf.nn.conv1d(attention_weights, F,
                             stride=1, padding='SAME',
                             #  use_cudnn_on_gpu=None,
                             #  data_format=None,
                             name='conv_features')
            W_fil = tf.contrib.layers.fully_connected(
                f,
                num_outputs=self.num_units,
                activation_fn=None,
                weights_initializer=tf.truncated_normal_initializer(
                    stddev=self.parameter_init),
                biases_initializer=tf.zeros_initializer(),
                scope="W_filter")

            v_a = tf.get_variable(
                "v_a", shape=[self.num_units], dtype=tf.float32)

            # Calculates a batch- and time-wise dot product with a variable
            energy = v_a * \
                tf.tanh(W_enc + tf.expand_dims(W_dec, axis=1) + W_fil)

        elif attention_type != 'location':
            ############################################################
            # location-based attention
            # f = F * α_{i-1}
            # energy = dot(v_a, tanh(W_dec(hidden_dec) + W_fil(f)))
            ############################################################
            with tf.control_dependencies(None):
                F = tf.Variable(tf.truncated_normal(
                    shape=[200, 1, 10],
                    # shape=[100, 1, 10],
                    stddev=0.1), name='filter')
            # `[B, T_in]` -> `[B, T_in, 1]`
            attention_weights = tf.expand_dims(attention_weights, axis=2)
            f = tf.nn.conv1d(attention_weights, F,
                             stride=1, padding='SAME',
                             #  use_cudnn_on_gpu=None,
                             #  data_format=None,
                             name='conv_features')
            W_fil = tf.contrib.layers.fully_connected(
                f,
                num_outputs=self.num_units,
                activation_fn=None,
                weights_initializer=tf.truncated_normal_initializer(
                    stddev=self.parameter_init),
                biases_initializer=tf.zeros_initializer(),
                scope="W_filter")

            v_a = tf.get_variable(
                "v_a", shape=[self.num_units], dtype=tf.float32)

            # Calculates a batch- and time-wise dot product with a variable
            energy = v_a * tf.tanh(tf.expand_dims(W_dec, axis=1) + W_fil)

        elif attention_type == 'luong_dot':
            raise NotImplementedError

        elif attention_type == 'luong_general':
            raise NotImplementedError

        elif attention_type == 'luong_concat':
            raise NotImplementedError

        elif attention_type == 'baidu_attetion':
            raise NotImplementedError

        # Sum by ?? axis (e_ij) (j=0,1,...)
        attention_sum = tf.reduce_sum(energy, axis=2)
        # energy: `[B, T_in, num_units]`
        # TODO: axis=1ではない？

        return attention_sum
