#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""A basic sequence decoder that performs a softmax based on the RNN state."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import tensorflow as tf
from tensorflow.python.util import nest

from models.attention.decoders.dynamic_decoder import dynamic_decode


class AttentionDecoderOutput(namedtuple(
        "DecoderOutput",
        [
            "logits",
            "predicted_ids",
            "decoder_output",
            "attention_weights",
            "context_vector"
        ])):
    pass


class AttentionDecoder(tf.contrib.seq2seq.Decoder):
    """An RNN Decoder that uses attention over an input sequence.
    Args:
        rnn_cell: An instance of ` tf.contrib.rnn.RNNCell`
        parameter_init (float): Range of uniform distribution to
            initialize weight parameters
        max_decode_length (int): the length of output sequences to stop
            prediction when EOS token have not been emitted
        num_classes (int): Output vocabulary size,
             i.e. number of units in the softmax layer
        encoder_outputs: The outputs of the encoder. A tensor of shape
            `[B, T_in, encoder_num_units]`.
        encoder_outputs_seq_len: Sequence length of the encoder outputs.
            An int32 Tensor of shape `[B]`.
        attention_layer: The attention function to use. This function map from
            `(state, inputs)` to `(attention_weights, context_vector)`.
            For an example, see `decoders.attention_layer.AttentionLayer`.
        time_major (bool): if True, time-major computation will be performed
            in the dynamic decoder.
        mode: tf.contrib.learn.ModeKeys
    """

    def __init__(self,
                 rnn_cell,
                 parameter_init,
                 max_decode_length,
                 num_classes,
                 encoder_outputs,
                 encoder_outputs_seq_len,
                 attention_layer,
                 time_major,
                 mode,
                 name='attention_decoder'):

        super(AttentionDecoder, self).__init__()

        self.rnn_cell = rnn_cell
        self.parameter_init = parameter_init
        self.max_decode_length = max_decode_length
        self.num_classes = num_classes
        self.encoder_outputs = encoder_outputs
        self.encoder_outputs_seq_len = encoder_outputs_seq_len
        self.attention_layer = attention_layer  # AttentionLayer class
        self.time_major = time_major
        self.mode = mode
        self.reuse = False if mode == tf.contrib.learn.ModeKeys.TRAIN else True
        self.name = name

        # NOTE: Not initialized yet
        self.initial_state = None
        self.helper = None

    @property
    def output_size(self):
        return AttentionDecoderOutput(
            logits=self.num_classes,
            predicted_ids=tf.TensorShape([]),
            decoder_output=self.rnn_cell.output_size,
            attention_weights=tf.shape(self.encoder_outputs)[1:-1],
            context_vector=self.encoder_outputs.get_shape()[-1])

    @property
    def output_dtype(self):
        return AttentionDecoderOutput(
            logits=tf.float32,
            predicted_ids=tf.int32,
            decoder_output=tf.float32,
            attention_weights=tf.float32,
            context_vector=tf.float32)

    @property
    def batch_size(self):
        return tf.shape(nest.flatten([self.initial_state])[0])[0]

    def __call__(self, initial_state, helper):
        """
        Args:
            initial_state: A tensor or tuple of tensors used as the initial
                rnn_cell state. Set to the final state of the encoder by default.
            helper: An instance of `tf.contrib.seq2seq.Helper` to assist
                decoding
        Returns:
            A tuple of `(outputs, final_state)`
                outputs: A tensor of `[T_out, B, num_classes]`
                final_state: A tensor of `[T_out, B, decoder_num_units]`
        """
        with tf.variable_scope('attention_decoder', reuse=self.reuse):
            # Initialize
            if self.initial_state is None:
                self._setup(initial_state, helper)
            # NOTE: ignore when attention_decoder is wrapped by
            # beam_search_decoder

            if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
                maximum_iterations = None
            else:
                maximum_iterations = self.max_decode_length

            # outputs, final_state, final_seq_len =
            # tf.contrib.seq2seq.dynamic_decode(
            outputs, final_state = dynamic_decode(
                decoder=self,
                output_time_major=self.time_major,
                impute_finished=True,
                maximum_iterations=maximum_iterations,
                scope=None)

            # tf.contrib.seq2seq.dynamic_decode
            # return self.finalize(outputs, final_state, final_seq_len)

            # ./dynamic_decoder.py
            return self.finalize(outputs, final_state, None)

    def initialize(self):
        """Initialize the decoder.
        Returns:
            finished: A tensor of size `[B, ]`
            first_inputs: The first inputs to the decoder. A tensor of size
                `[B, embedding_dim]`
            initial_state: The initial decoder state. A tensor of size
                `[B, ]`
        """
        # Create inputs for the first time step
        finished, first_inputs = self.helper.initialize()
        # finished: `[1]`
        # first_inputs: `[B]`

        # Concat empty attention context (Input-feeding approach)
        batch_size = tf.shape(first_inputs)[0]
        encoder_num_units = self.encoder_outputs.get_shape().as_list()[-1]
        context_vector = tf.zeros(shape=[batch_size, encoder_num_units])
        first_inputs = tf.concat([first_inputs, context_vector], axis=1)

        # Initialize attention weights
        self.attention_weights = tf.zeros(
            shape=[batch_size, tf.shape(self.encoder_outputs)[1]])
        # `[B, T_in]`

        return finished, first_inputs, self.initial_state
        # TODO: self.initial_stateはこの時点ではNone??

    def _compute_output(self, decoder_output, attention_weights):
        """Computes the decoder outputs at each time.
        Args:
            decoder_output: The previous state of the decoder
            attention_weights: A tensor of size `[B, ]`
        Returns:
            attentional_vector: A tensors of size `[B, ]`
            logits: A tensor of size `[B, ]`
            attention_weights: A tensor of size `[B, ]`
            context_vector: A tensor of szie `[B, ]`
        """
        # Compute attention weights & context vector
        attention_weights, context_vector = self.attention_layer(
            encoder_outputs=self.encoder_outputs,
            decoder_output=decoder_output,
            encoder_outputs_length=self.encoder_outputs_seq_len,
            attention_weights=attention_weights)

        # Input-feeding approach, this is used as inputs for the decoder
        attentional_vector = tf.contrib.layers.fully_connected(
            tf.concat([decoder_output, context_vector], axis=1),
            num_outputs=self.rnn_cell.output_size,
            activation_fn=tf.nn.tanh,
            weights_initializer=tf.truncated_normal_initializer(
                stddev=self.parameter_init),
            biases_initializer=None,  # no bias
            scope="attentional_vector")
        # NOTE: This makes the softmax smaller and allows us to synthesize
        # information between decoder state and attention context
        # see https://arxiv.org/abs/1508.04025v5

        # Softmax computation
        logits = tf.contrib.layers.fully_connected(
            attentional_vector,
            num_outputs=self.num_classes,
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(
                stddev=self.parameter_init),
            biases_initializer=tf.zeros_initializer(),
            scope="output_layer")

        return attentional_vector, logits, attention_weights, context_vector

    def _setup(self, initial_state, helper):
        """Sets the initial state and helper for the decoder.
        Args:
            initial_state:
            helper:
        """
        self.initial_state = initial_state
        self.helper = helper

        def _att_next_inputs(time, outputs, state, sample_ids, name=None):
            """Wraps the original decoder helper function to append the
               attention context.
            Args:
                time:
                outputs:
                state:
                sample_ids:
                name (string, optional): Name scope for any created operations
            Returs:
                Returns:
                    finished: A tensor of size `[B, ]`
                    next_inputs: A tensor of size `[B, ]`
                    next_state: A tensor of size `[B, ]`
            """
            finished, next_inputs, next_state = helper.next_inputs(
                time=time,
                outputs=outputs,
                state=state,
                sample_ids=sample_ids,
                name=name)

            # Input-feeding approach
            next_inputs = tf.concat(
                [next_inputs, outputs.context_vector], axis=1)

            return finished, next_inputs, next_state

        # Wrap helper function for the attention mechanism
        self.helper = tf.contrib.seq2seq.CustomHelper(
            initialize_fn=helper.initialize,
            sample_fn=helper.sample,
            next_inputs_fn=_att_next_inputs)

    def step(self, time, inputs, state, name=None):
        """Perform a decoding step.
        Args:
           time:
           inputs:
           state:
           name (string, optional): Name scope for any created operations
        Returns:
            outputs: An instance of AttentionDecoderOutput
            next_state: A state tensors and TensorArrays
            next_inputs: The tensor that should be used as input for the
                next step
            finished: A boolean tensor telling whether the sequence is
                complete, for each sequence in the batch
        """
        cell_output, cell_state = self.rnn_cell(inputs, state)
        attentional_vector, logits, attention_weights, context_vector = self._compute_output(
            decoder_output=cell_output,
            attention_weights=self.attention_weights)

        # Update attention weights
        self.attention_weights = attention_weights

        sample_ids = self.helper.sample(time=time,
                                        outputs=logits,
                                        state=cell_state)

        outputs = AttentionDecoderOutput(logits=logits,
                                         predicted_ids=sample_ids,
                                         decoder_output=attentional_vector,
                                         attention_weights=attention_weights,
                                         context_vector=context_vector)

        finished, next_inputs, next_state = self.helper.next_inputs(
            time=time,
            outputs=outputs,
            state=cell_state,
            sample_ids=sample_ids)

        return outputs, next_state, next_inputs, finished

    def finalize(self, outputs, final_state, final_seq_len):
        """Applies final transformation to the decoder output once decoding is
           finished.
        Args:
            outputs:
            final_state:
            final_seq_len:
        Returns:
            outputs:
            final_state:
        """
        return outputs, final_state
