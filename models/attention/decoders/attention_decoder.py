#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""A basic sequence decoder that performs a softmax based on the RNN state."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import tensorflow as tf
from tensorflow.python.util import nest
from .dynamic_decoder import dynamic_decode


# template
AttentionDecoderOutput = namedtuple(
    "DecoderOutput",
    [
        "logits",
        "predicted_ids",
        "cell_output",
        "attention_scores",
        "attention_context"
    ])


class AttentionDecoder(tf.contrib.seq2seq.Decoder):
    """An RNN Decoder that uses attention over an input sequence.
    Args:
        cell: An instance of ` tf.contrib.rnn.RNNCell` (LSTM, GRU is also OK)
        parameter_init: A float value. Range of uniform distribution to
            initialize weight parameters
        max_decode_length:
        num_classes: Output vocabulary size,
             i.e. number of units in the softmax layer
        attention_encoder_states: The sequence used to calculate attention
            scores. A tensor of shape
            `[batch_size, max_time, encoder_num_units]`.
        attention_values: The sequence to attend over.
            A tensor of shape `[batch_size, max_time, encoder_num_units]`.
        attention_values_length: Sequence length of the attention values.
            An int32 Tensor of shape `[batch]`.
        attention_layer: The attention function to use. This function map from
            `(state, inputs)` to `(attention_weights, attention_context)`.
            For an example, see `decoders.attention_layer.AttentionLayer`.
    """

    def __init__(self,
                 cell,
                 parameter_init,
                 max_decode_length,
                 num_classes,
                 attention_encoder_states,
                 attention_values,
                 attention_values_length,
                 attention_layer,
                 name='attention_decoder'):
        # param
        self.cell = cell
        self.parameter_init = parameter_init
        self.max_decode_length = max_decode_length

        self.num_classes = num_classes
        self.attention_encoder_states = attention_encoder_states
        self.attention_values = attention_values
        self.attention_values_length = attention_values_length
        self.attention_layer = attention_layer  # AttentionLayer class
        self.name = name

        # Not initialized yet
        self.initial_state = None
        self.helper = None

    def __call__(self, *args, **kwargs):
        # TODO: variable_scope
        return self._build(*args, **kwargs)

    @property
    def output_size(self):
        return AttentionDecoderOutput(
            logits=self.num_classes,
            predicted_ids=tf.TensorShape([]),
            cell_output=self.cell.output_size,
            attention_scores=tf.shape(self.attention_values)[1:-1],
            attention_context=self.attention_values.get_shape()[-1])

    @property
    def output_dtype(self):
        return AttentionDecoderOutput(
            logits=tf.float32,
            predicted_ids=tf.int32,
            cell_output=tf.float32,
            attention_scores=tf.float32,
            attention_context=tf.float32)

    @property
    def batch_size(self):
        return tf.shape(nest.flatten([self.initial_state])[0])[0]

    def _build(self, initial_state, helper, mode):
        """
        Args:
            helper: An instance of `tf.contrib.seq2seq.Helper` to assist
                decoding
            initial_state: A tensor or tuple of tensors used as the initial
                cell state. Set to the final state of the encoder by default.
            mode:
        Returns:
            A tuple of `(outputs, final_state)`
        """
        self.mode = mode
        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.reuse = False
        else:
            self.reuse = True

        # Initialize
        if not self.initial_state:
            self._setup(initial_state, helper)

        scope = tf.get_variable_scope()
        scope.set_initializer(tf.random_uniform_initializer(
            -self.parameter_init,
            self.parameter_init))

        maximum_iterations = None
        if self.mode == tf.contrib.learn.ModeKeys.INFER:
            maximum_iterations = self.max_decode_length

        # outputs, final_state = tf.contrib.seq2seq.dynamic_decode(
        #     decoder=self,
        #     output_time_major=False,  # changed
        #     impute_finished=True,  # changed
        #     maximum_iterations=maximum_iterations)
        outputs, final_state = dynamic_decode(
            decoder=self,
            output_time_major=False,  # changed
            impute_finished=False,  # changed
            maximum_iterations=maximum_iterations)
        return self.finalize(outputs, final_state)

    def finalize(self, outputs, final_state):
        """Applies final transformation to the decoder output once decoding is
           finished.
        Args: outputs:
              final_state:
        Returns:
            A tuple of `(outputs, final_state)`
        """
        return (outputs, final_state)

    def initialize(self, name=None):
        """
        Args:
            name:
        Returns:
            finished:
            first_inputs:
            initial_state:
        """
        # Create inputs for the first time step
        finished, first_inputs = self.helper.initialize()
        print(first_inputs.get_shape().as_list())

        # first_inputs: `[batch_size, embedding_dim]`

        # Concat empty attention context
        batch_size = tf.shape(first_inputs)[0]
        encoder_num_units = self.attention_values.get_shape().as_list()[-1]
        attention_context = tf.zeros(shape=[batch_size, encoder_num_units])

        # Create first inputs
        first_inputs = tf.concat([first_inputs, attention_context], axis=1)
        # ex.)
        # tensor t3 with shape [2, 3]
        # tensor t4 with shape [2, 3]
        # tf.shape(tf.concat([t3, t4], 0)) ==> [4, 3]
        # tf.shape(tf.concat([t3, t4], 1)) ==> [2, 6]

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
        attention_weights, attention_context = self.attention_layer(
            encoder_states=self.attention_encoder_states,
            current_decoder_state=cell_output,
            values=self.attention_values,
            values_length=self.attention_values_length)

        # TODO: Make this a parameter: We may or may not want this.
        # Transform attention context.
        # This makes the softmax smaller and allows us to synthesize
        # information between decoder state and attention context
        # see https://arxiv.org/abs/1508.04025v5
        # g_i = tanh(W_s * s_{i-1} + W_c * c_i + b (+ W_o * y_{i-1}))
        self.softmax_input = tf.contrib.layers.fully_connected(
            inputs=tf.concat([cell_output, attention_context], axis=1),
            num_outputs=self.cell.output_size,
            activation_fn=tf.nn.tanh,
            # reuse=True,
            scope="attention_mix")
        # TODO: y_i-1も入力にするのは冗長らしいが，自分で確かめる

        # Softmax computation
        # P(y_i|s_i, c_i, y_{i-1}) = softmax(W_g * g_i + b)
        logits = tf.contrib.layers.fully_connected(
            inputs=self.softmax_input,
            num_outputs=self.num_classes,
            activation_fn=None,
            # reuse=True,
            scope="logits")
        self.logits = logits

        return (self.softmax_input, logits,
                attention_weights, attention_context)

    def _setup(self, initial_state, helper):
        self.initial_state = initial_state
        self.helper = helper

        def att_next_inputs(time, outputs, state, sample_ids, name=None):
            """Wraps the original decoder helper function to append the
               attention context.
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
                [next_inputs, outputs.attention_context], axis=1)

            return (finished, next_inputs, next_state)

        self.helper = tf.contrib.seq2seq.CustomHelper(
            initialize_fn=helper.initialize,
            sample_fn=helper.sample,
            next_inputs_fn=att_next_inputs)

    def step(self, time, inputs, state, name=None):
        """Perform a decoding step.
        Args:
           time: scalar `int32` tensor.
           inputs: A input tensors.
           state: A state tensors and TensorArrays.
           name: Name scope for any created operations.
        Returns:
            A tuple of `(outputs, naxt_state, next_inputs, finished)`
                outputs: An instance of AttentionDecoderOutput
                next_state: A state tensors and TensorArrays
                next_inputs: The tensor that should be used as input for the
                    next step
                finished: A boolean tensor telling whether the sequence is
                    complete, for each sequence in the batch.
        """
        with tf.variable_scope("step", reuse=self.reuse):
            # Call LSTMCell
            # print(inputs.get_shape().as_list())
            cell_output_prev, cell_state_prev = self.cell(inputs, state)
            cell_output, logits, attention_scores, attention_context = \
                self.compute_output(cell_output_prev)

            sample_ids = self.helper.sample(time=time,
                                            outputs=logits,
                                            state=cell_state_prev)
            # TODO: Trainingのときlogitsの値はone-hotまたは一意のベクトルに変換されているか？

            outputs = AttentionDecoderOutput(logits=logits,
                                             predicted_ids=sample_ids,
                                             cell_output=cell_output,
                                             attention_scores=attention_scores,
                                             attention_context=attention_context)

            finished, next_inputs, next_state = self.helper.next_inputs(
                time=time,
                outputs=outputs,
                state=cell_state_prev,
                sample_ids=sample_ids)

            return (outputs, next_state, next_inputs, finished)
