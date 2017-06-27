#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Attention layers for calculating attention weights."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class AttentionLayer(object):
    """Attention layer. This implementation is based on
        https://arxiv.org/abs/1409.0473.
            Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio.
            "Neural machine translation by jointly learning to align and
            translate."
            arXiv preprint arXiv:1409.0473 (2014).
    Args:
        num_unit: Number of units used in the attention layer
        attention_smoothing: bool, if True, replace exp to sigmoid function in
            the softmax layer of computing attention weights
        attention_weights_tempareture: A float value,
        attention_type: string, content or location or hybrid or layer_dot
    """

    def __init__(self, num_unit, attention_smoothing,
                 attention_weights_tempareture,
                 attention_type, name='attention_layer'):
        self.num_unit = num_unit
        self.attention_smoothing = attention_smoothing
        self.attention_weights_tempareture = attention_weights_tempareture
        self.attention_type = attention_type
        self.name = name

    def __call__(self, *args, **kwargs):
        # TODO: variable_scope
        return self._build(*args, **kwargs)

    def _build(self, encoder_states, current_decoder_state, values,
               values_length, attention_weights):
        """Computes attention scores and outputs.
        Args:
            encoder_states: The outputs of the encoder and equivalent to
                `values`. This is used to calculate attention scores.
                A tensor of shape `[batch_size, time, encoder_num_units]`
                where each element in the `time` dimension corresponds to the
                decoder states for that value.
            current_decoder_state: The current state of the docoder.
                This is used to calculate attention scores.
                A tensor of shape `[batch_size, decoder_num_units]`
            values: The sequence of encoder outputs to compute attention over.
                A tensor of shape `[batch_size, time, encoder_num_units]`.
            values_length: An int32 tensor of shape `[batch_size]` defining
                the sequence length of the attention values.
            attention_weights:
        Returns:
            A tuple `(attention_weights, attention_context)`.
                `attention_weights` is vector of length `time` where each
                    element is the normalized "score" of the corresponding
                    `inputs` element.
                    A tensor of shape `[batch_size, input_time]`
                `attention_context` is the final attention layer output
                    corresponding to the weighted inputs.
                    A tensor of shape `[batch_size, encoder_num_units]`.
        """
        # Compute attention scores over encoder outputs (energy: e_ij)
        # e_ij = f(V * h_j,  W * s_{i-1}, (U * f_ij))
        energy = self.attention_score_func(encoder_states,
                                           current_decoder_state,
                                           attention_weights)

        # Replace all scores for padded inputs with tf.float32.min
        num_scores = tf.shape(energy)[1]  # max_time
        scores_mask = tf.sequence_mask(
            lengths=tf.to_int32(values_length),
            maxlen=tf.to_int32(num_scores),
            dtype=tf.float32)
        # ex.) tf.sequence_mask
        # tf.sequence_mask([1, 3, 2], 5) = [[True, False, False, False, False],
        #                                   [True, True, True, False, False],
        #                                   [True, True, False, False, False]]
        energy = energy * scores_mask + ((1.0 - scores_mask) * tf.float32.min)
        # TODO: For underflow?

        energy /= self.attention_weights_tempareture

        # Normalize the scores (attention_weights: α_ij (j=0,1,...))
        if self.attention_smoothing:
            attention_weights = tf.sigmoid(energy) / tf.reduce_sum(
                tf.sigmoid(energy),
                axis=-1,
                keep_dims=True)
        else:
            # attention_weights = tf.nn.softmax(
            #     scores, name="attention_weights")
            attention_weights = tf.exp(energy) / tf.reduce_sum(
                tf.exp(energy),
                axis=-1,
                keep_dims=True)

        # Calculate the weighted average of the attention inputs
        # according to the scores
        # c_i = sum_{j}(α_ij * h_j)
        attention_context = tf.expand_dims(
            attention_weights, axis=2) * values
        attention_context = tf.reduce_sum(
            attention_context, axis=1, name="attention_context")
        values_depth = values.get_shape().as_list()[-1]  # = encoder_num_units
        # `[batch_size, encoder_num_units]`
        attention_context.set_shape([None, values_depth])

        return (attention_weights, attention_context)

    def attention_score_func(self, encoder_states, current_decoder_state,
                             attention_weights):
        """An attention layer that calculates attention scores.
        Args:
            encoder_states: The sequence of encoder outputs
                A tensor of shape `[batch_size, input_time, encoder_num_units]`
            current_decoder_state: The current state of the docoder
                A tensor of shape `[batch_size, decoder_num_units]`
            attention_weights: A tensor of size `[batch_size, input_time]`
        Returns:
            attention_sum: The summation of attention scores
                A tensor of shape `[batch_size, input_time]`
        """
        # with tf.variable_scope(self.attention_type):

        # Fully connected layers to transform both encoder_states and
        # current_decoder_state into a tensor with `num_unit` units
        # W * s_{i-1} (i: output time index)
        Ws = tf.contrib.layers.fully_connected(
            inputs=current_decoder_state,
            num_outputs=self.num_unit,
            activation_fn=None,
            # reuse=True,
            scope="Ws")

        if self.attention_type != 'location':
            # V * h_j (j: input time index)
            Vh = tf.contrib.layers.fully_connected(
                inputs=encoder_states,
                num_outputs=self.num_unit,
                activation_fn=None,
                # reuse=True,
                scope="Vh")
        # NOTE: Bias terms are already included in these layers

        if self.attention_type == 'content':
            ############################################################
            # content-based attention
            # e_ij = wT * tanh(W * s_{i-1} + V * h_j + bias)
            ############################################################
            w = tf.get_variable(
                "w", shape=[self.num_unit], dtype=tf.float32)

            # Calculates a batch- and time-wise dot product with a variable
            e_ij = w * tf.tanh(tf.expand_dims(Ws, axis=1) + Vh)
            # ex.) tf.expand_dims
            # 't2' is a tensor of shape [2, 3, 5]
            # tf.shape(expand_dims(t2, 0)) ==> [1, 2, 3, 5]
            # tf.shape(expand_dims(t2, 2)) ==> [2, 3, 1, 5]
            # tf.shape(expand_dims(t2, 3)) ==> [2, 3, 5, 1]

        elif self.attention_type == 'location':
            ############################################################
            # location-based attention
            # e_ij = wT * tanh(W * s_{i-1} + U * f_ij + bias)
            ############################################################
            F = tf.Variable(tf.truncated_normal(
                shape=[100, 1, 10], stddev=0.1, name='Filter'))
            f = tf.nn.conv1d(tf.expand_dims(attention_weights, axis=2), F,
                             stride=1, padding='SAME',
                             #  use_cudnn_on_gpu=None,
                             #  data_format=None,
                             name='f')

            # U * f_ij
            Uf = tf.contrib.layers.fully_connected(
                inputs=f,
                num_outputs=self.num_unit,
                activation_fn=None,
                # reuse=True,
                scope="Uf")

            w = tf.get_variable(
                "w", shape=[self.num_unit], dtype=tf.float32)

            # Calculates a batch- and time-wise dot product with a variable
            e_ij = w * tf.tanh(tf.expand_dims(Ws, axis=1) + Uf)

        elif self.attention_type == 'hybrid':
            ############################################################
            # hybrid attention (content-based + location-based)
            # e_ij = wT * tanh(W * s_{i-1} + V * h_j + U * f_ij + bias)
            ############################################################
            F = tf.Variable(tf.truncated_normal(
                shape=[100, 1, 10], stddev=0.1, name='Filter'))
            f = tf.nn.conv1d(tf.expand_dims(attention_weights, axis=2), F,
                             stride=1, padding='SAME',
                             #  use_cudnn_on_gpu=None,
                             #  data_format=None,
                             name='f')

            # U * f_ij
            Uf = tf.contrib.layers.fully_connected(
                inputs=f,
                num_outputs=self.num_unit,
                activation_fn=None,
                # reuse=True,
                scope="Uf")

            w = tf.get_variable(
                "w", shape=[self.num_unit], dtype=tf.float32)

            # Calculates a batch- and time-wise dot product with a variable
            e_ij = w * tf.tanh(tf.expand_dims(Ws, axis=1) + Vh + Uf)

        elif self.attention_type == 'layer_dot':
            ############################################################
            # layer_dot attention
            # e_ij = (W * s_{i-1}) * (U * f_ij) + bias
            ############################################################
            # Calculates a batch- and time-wise dot product
            e_ij = tf.expand_dims(Ws, axis=1) * Vh

        else:
            raise ValueError(
                'attention_type is "content" or "location" or ' +
                '"hybrid" or "layer_dot".')

        # Sum by num_unit axis (e_ij) (j=0,1,...)
        attention_sum = tf.reduce_sum(e_ij, axis=2)
        # e_ij: `[batch_size, input_time(j), num_unit]`

        return attention_sum
