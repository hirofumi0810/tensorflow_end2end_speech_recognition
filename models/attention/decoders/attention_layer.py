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
        num_units: Number of units used in the attention layer
        attention_type: bahdanau or layer_dot
    """

    def __init__(self, num_units, attention_type='bahdanau',
                 name='attention_layer'):
        self.num_units = num_units
        self.attention_type = attention_type
        self.name = name

    def __call__(self, *args, **kwargs):
        # TODO: variable_scope
        return self._build(*args, **kwargs)

    def _build(self, encoder_states, current_decoder_state, values,
               values_length):
        """Computes attention scores and outputs.
        Args:
            encoder_states: The outputs of the encoder and equivalent to
                `values`. This is used to calculate attention scores.
                A tensor of shape `[batch_size, max_time, encoder_num_units]`
                where each element in the `time` dimension corresponds to the
                decoder states for that value.
            current_decoder_state: The current state of the docoder.
                This is used to calculate attention scores.
                A tensor of shape `[batch_size, decoder_num_units]`
            values: The sequence of encoder outputs to compute attention over.
                A tensor of shape `[batch_size, max_time, encoder_num_units]`.
            values_length: An int32 tensor of shape `[batch_size]` defining
                the sequence length of the attention values.
        Returns:
            A tuple `(attention_weights, attention_context)`.
                `attention_weights` is vector of length `time` where each
                    element is the normalized "score" of the corresponding
                    `inputs` element.
                    A tensor of shape `[batch_size, max_time]`
                `attention_context` is the final attention layer output
                    corresponding to the weighted inputs.
                    A tensor of shape `[batch_size, encoder_num_units]`.
        """
        # Fully connected layers to transform both encoder_states and
        # current_decoder_state into a tensor with `num_units` units
        # h_j (j: time index of input) => U_a * h_j
        att_encoder_states = tf.contrib.layers.fully_connected(
            inputs=encoder_states,
            num_outputs=self.num_units,
            activation_fn=None,
            # reuse=True,
            scope="att_encoder_states")

        # s_{i-1} (i: time index of output) => W_a * s_{i-1}
        att_decoder_state = tf.contrib.layers.fully_connected(
            inputs=current_decoder_state,
            num_outputs=self.num_units,
            activation_fn=None,
            # reuse=True,
            scope="att_decoder_state")
        # TODO: Divide self.num_units into encoder_num_units and
        # decoder_num_units
        # NOTE: エンコーダがBidirectionalのときユニット数を2倍にすることに注意

        # Compute attention scores over encoder outputs (energy: e_ij)
        # v_a = f(U_a * h_j, W_a * s_{i-1})
        scores = self.attention_score_func(att_encoder_states,
                                           att_decoder_state)

        # Replace all scores for padded inputs with tf.float32.min
        num_scores = tf.shape(scores)[1]  # input length
        scores_mask = tf.sequence_mask(
            lengths=tf.to_int32(values_length),
            maxlen=tf.to_int32(num_scores),
            dtype=tf.float32)
        # ex.)
        # tf.sequence_mask([1, 3, 2], 5) = [[True, False, False, False, False],
        #                                   [True, True, True, False, False],
        #                                   [True, True, False, False, False]]
        scores = scores * scores_mask + ((1.0 - scores_mask) * tf.float32.min)
        # TODO: For underflow?

        # Normalize the scores (attention_weights: α_ij (j=0,1,...))
        attention_weights = tf.nn.softmax(scores, name="attention_weights")
        # TODO: Add beta(temperature) in order to smooth output probabilities

        # Calculate the weighted average of the attention inputs
        # according to the scores
        # c_i = sigma_{j}(α_ij * h_j)
        attention_context = tf.expand_dims(attention_weights, axis=2) * values
        attention_context = tf.reduce_sum(
            attention_context, axis=1, name="attention_context")
        values_depth = values.get_shape().as_list()[-1]  # = encoder_num_units
        # `[batch_size, encoder_num_units]`
        attention_context.set_shape([None, values_depth])

        return (attention_weights, attention_context)

    def attention_score_func(self, encoder_states, current_decoder_state):
        """An attention layer that calculates attention scores.
        Args:
            encoder_states: The sequence of encoder outputs
                A tensor of shape `[batch_size, max_time, encoder_num_units]`
            current_decoder_state: The current state of the docoder
                A tensor of shape `[batch_size, decoder_num_units]`
        Returns:
            attention_sum: The summation of attention scores (energy: e_ij)
            A tensor of shape `[batch_size, max_time, ?]`
        """
        if self.attention_type == 'bahdanau':
            # with tf.variable_scope("bahdanau", reuse=True):
            v_att = tf.get_variable("v_att",
                                    shape=[self.num_units],
                                    dtype=tf.float32)

            # calculates a batch- and time-wise dot product with a variable
            # v_a = tanh(W_a * s_{i-1} + U_a * h_j)
            attention_sum = tf.reduce_sum(v_att * tf.tanh(
                tf.expand_dims(current_decoder_state, axis=1) +
                encoder_states), [2])
            # ex.)
            # 't2' is a tensor of shape [2, 3, 5]
            # tf.shape(expand_dims(t2, 0)) ==> [1, 2, 3, 5]
            # tf.shape(expand_dims(t2, 2)) ==> [2, 3, 1, 5]
            # tf.shape(expand_dims(t2, 3)) ==> [2, 3, 5, 1]

            # TODO: what does [2] mean? axis?

        elif self.attention_type == 'layer_dot':
            # calculates a batch- and time-wise dot product
            attention_sum = tf.reduce_sum(
                tf.expand_dims(current_decoder_state, axis=1) *
                encoder_states, [2])

        else:
            # TODO: Add other versions
            raise ValueError('attention_type is "bahdanau" or "layer_dot".')

        return attention_sum
