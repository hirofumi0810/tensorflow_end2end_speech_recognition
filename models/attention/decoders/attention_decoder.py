#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""A basic sequence decoder that performs a softmax based on the RNN state."""

from collections import namedtuple
import tensorflow as tf
from helper.custom_helper import CustomHelper


# template
AttentionDecoderOutput = namedtuple(
    "DecoderOutput",
    "logits predicted_ids cell_output attention_scores attention_context")


class AttentionDecoder(object):
    """An RNN Decoder that uses attention over an input sequence.
    Args:
        cell: An instance of ` tf.contrib.rnn.RNNCell` (LSTM, GRU is OK)
        helper: An instance of `tf.contrib.seq2seq.Helper` to assist decoding
        initial_state: A tensor or tuple of tensors used as the initial cell state.
            Set to the final state of the encoder.
        num_classes: Output vocabulary size,
             i.e. number of units in the softmax layer
        attention_encoder_states: The sequence used to calculate attention scores.
            A tensor of shape `[batch, time, encoder_num_units]`.
        attention_values: The sequence to attend over.
            A tensor of shape `[batch, time, encoder_num_units]`.
        attention_values_length: Sequence length of the attention values.
            An int32 Tensor of shape `[batch]`.
        attention_func: The attention function to use. This function map from
            `(state, inputs)` to `(attention_weights, attention_context)`.
            For an example, see `decoders.attention_layer.AttentionLayer`.
    """

    def __init__(self,
                 cell,
                 helper,
                 initial_state,
                 num_classes,
                 attention_encoder_states,
                 attention_values,
                 attention_values_length,
                 attention_func,
                 name="attention_decoder"):
        self.cell = cell
        # self.helper = helper
        self.initial_state = initial_state
        self.num_classes = num_classes
        self.attention_encoder_states = attention_encoder_states
        self.attention_values = attention_values
        self.attention_values_length = attention_values_length
        self.attention_func = attention_func  # AttentionLayer class

    def initialize(self, name=None):
        """
        Args:
            name:
        Returns:
            finished:
            first_inputs:
            initial_state:
        """
        finished, first_inputs = self.helper.initialize()
        # TODO: what is self.helper.initialize()

        # Concat empty attention context
        attention_context = tf.zeros([
            tf.shape(first_inputs)[0],
            self.attention_values.get_shape().as_list()[-1]
        ])
        first_inputs = tf.concat([first_inputs, attention_context], axis=1)

        return finished, first_inputs, self.initial_state

    def compute_output(self, cell_output):
        """Computes the decoder outputs at each time.
        Args:
            cell_output: The previous state of the decoder
        Returns:
            softmax_input:
            logits:
            attention_weights:
            attention_context:
        """
        # Compute attention weights & context
        attention_weights, attention_context = self.attention_func(
            encoder_states=self.attention_encoder_states,
            current_decoder_state=cell_output,
            values=self.attention_values,
            values_length=self.attention_values_length)

        # TODO: Make this a parameter: We may or may not want this.
        # Transform attention context.
        # This makes the softmax smaller and allows us to synthesize information
        # between decoder state and attention context
        # see https://arxiv.org/abs/1508.04025v5
        # t_i = U_o * s_{i-1} + C_o * c_i (+ V_o * E * y_{i-1})
        softmax_input = tf.contrib.layers.fully_connected(
            inputs=tf.concat([cell_output, attention_context], axis=1),
            num_outputs=self.cell.output_size,
            activation_fn=tf.tanh,
            scope="attention_mix")
        # TODO: y_i-1も入力にするのは冗長らしいが，自分で確かめる
        # TODO: why tf.tanh?

        # Softmax computation
        # y_i = g(s_{i-1},c_i)
        # P(y_i|s_i, y_{i-1}, t_i) ∝ exp(y_i^T * W_o * t_i)
        logits = tf.contrib.layers.fully_connected(
            inputs=softmax_input,
            num_outputs=self.num_classes,
            activation_fn=None,
            scope="logits")

        return softmax_input, logits, attention_weights, attention_context

    def _setup(self, initial_state, helper):
        self.initial_state = initial_state

        def att_next_inputs(time, outputs, state, sample_ids, name=None):
            """Wraps the original decoder helper function to append the attention context.
            Args:
                time:
                outputs:
                state:
                sample_ids:
                name:
            Returs:
                A tuple of `(finished, next_inputs, next_state)`
            """
            finished, next_inputs, next_state = helper.next_inputs(
                time=time,
                outputs=outputs,
                state=state,
                sample_ids=sample_ids,
                name=name)
            next_inputs = tf.concat(
                [next_inputs, outputs.attention_context], 1)
            return (finished, next_inputs, next_state)

        self.helper = CustomHelper(
            initialize_fn=helper.initialize,
            sample_fn=helper.sample,
            next_inputs_fn=att_next_inputs)

    def step(self, time_, inputs, state, name=None):
        """
        Args:
            time_:
            inputs:
            state:
            name:
        Returns:
            A tuple of `(outputs, naxt_state, next_inputs, finished)`
                outputs:
                next_state:
                next_inputs:
                finished:
        """
        cell_output, cell_state = self.cell(inputs, state)
        cell_output_new, logits, attention_scores, attention_context = \
            self.compute_output(cell_output)

        sample_ids = self.helper.sample(
            time=time_, outputs=logits, state=cell_state)

        outputs = AttentionDecoderOutput(
            logits=logits,
            predicted_ids=sample_ids,
            cell_output=cell_output_new,
            attention_scores=attention_scores,
            attention_context=attention_context)

        finished, next_inputs, next_state = self.helper.next_inputs(
            time=time_, outputs=outputs, state=cell_state, sample_ids=sample_ids)

        return (outputs, next_state, next_inputs, finished)
