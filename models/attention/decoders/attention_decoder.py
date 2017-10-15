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
        time-major (bool): if True,
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
        self.name = name

        # NOTE: Not initialized yet
        self.initial_state = None
        self.helper = None

        self.reuse = True
        # NOTE: This is for beam search decoder
        # When training mode, this will be overwritten in self._build()

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

    def __call__(self, initial_state, helper, mode):
        """
        Args:
            initial_state: A tensor or tuple of tensors used as the initial
                rnn_cell state. Set to the final state of the encoder by default.
            helper: An instance of `tf.contrib.seq2seq.Helper` to assist
                decoding
            mode:
        Returns:
            A tuple of `(outputs, final_state)`
                outputs: A tensor of `[T_out, B, num_classes]`
                final_state: A tensor of `[T_out, B, decoder_num_units]`
        """
        # Initialize
        if self.initial_state is None:
            self._setup(initial_state, helper)
        # NOTE: ignore when attention_decoder is wrapped by beam_search_decoder

        # scope = tf.get_variable_scope()
        # scope.set_initializer(tf.random_uniform_initializer(
        #     minval=-self.parameter_init,
        #     maxval=self.parameter_init))

        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.reuse = False
            maximum_iterations = None
        else:
            self.reuse = True
            maximum_iterations = self.max_decode_length

        # outputs, final_state, final_seq_len =
        # tf.contrib.seq2seq.dynamic_decode(
        outputs, final_state = dynamic_decode(
            decoder=self,
            output_time_major=self.time_major,
            impute_finished=True,
            maximum_iterations=maximum_iterations,
            scope='dynamic_decoder')

        # tf.contrib.seq2seq.dynamic_decode
        # return self.finalize(outputs, final_state, final_seq_len)

        # ./dynamic_decoder.py
        return self.finalize(outputs, final_state, None)

    def initialize(self):
        """Initialize the decoder.
        Returns:
            finished: A tensor of size `[]`
            first_inputs: A tensor of size `[B, embedding_dim]`
            initial_state: A tensor of size `[]`
        """
        # Create inputs for the first time step
        finished, first_inputs = self.helper.initialize()
        # TODO: 最初のshapeをcheck

        # Concat empty attention context
        batch_size = tf.shape(first_inputs)[0]
        encoder_num_units = self.encoder_outputs.get_shape().as_list()[-1]
        context_vector = tf.zeros(shape=[batch_size, encoder_num_units])
        first_inputs = tf.concat([first_inputs, context_vector], axis=1)
        # TODO: これconcatしたらおかしくない？

        # Initialize attention weights
        self.attention_weights = tf.zeros(
            shape=[batch_size, tf.shape(self.encoder_outputs)[1]])
        # `[B, T_in]`

        return finished, first_inputs, self.initial_state
        # TODO: self.initial_stateはこの時点ではNone??

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

    def compute_output(self, decoder_output, attention_weights):
        """Computes the decoder outputs at each time.
        Args:
            decoder_output: The previous state of the decoder
            attention_weights:
        Returns:
            softmax_input: A tensor of size `[]`
            logits: A tensor of size `[]`
            attention_weights: A tensor of size `[]`
            context_vector: A tensor of szie `[]`
        """
        # Compute attention weights & context
        attention_weights, context_vector = self.attention_layer(
            encoder_outputs=self.encoder_outputs,
            decoder_output=decoder_output,
            encoder_outputs_length=self.encoder_outputs_seq_len,
            attention_weights=attention_weights)

        # TODO: Make this a parameter: We may or may not want this.
        # Transform attention context.
        # This makes the softmax smaller and allows us to synthesize
        # information between decoder state and attention context
        # see https://arxiv.org/abs/1508.04025v5
        proj = tf.contrib.layers.fully_connected(
            tf.concat([decoder_output, context_vector], axis=1),
            num_outputs=self.rnn_cell.output_size,
            activation_fn=tf.nn.tanh,
            weights_initializer=tf.truncated_normal_initializer(
                stddev=self.parameter_init),
            biases_initializer=tf.zeros_initializer(),
            scope="output_proj")

        # Softmax computation
        logits = tf.contrib.layers.fully_connected(
            proj,
            num_outputs=self.num_classes,
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(
                stddev=self.parameter_init),
            biases_initializer=tf.zeros_initializer(),
            scope="output_layer")
        self.logits = logits
        # 何に使う？

        return proj, logits, attention_weights, context_vector

    def _setup(self, initial_state, helper):
        """Define original helper function."""
        self.initial_state = initial_state
        self.helper = helper
        # これいらんくない？

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
                [next_inputs, outputs.context_vector], axis=1)

            return finished, next_inputs, next_state

        self.helper = tf.contrib.seq2seq.CustomHelper(
            initialize_fn=helper.initialize,
            sample_fn=helper.sample,
            next_inputs_fn=att_next_inputs)

    def step(self, time, inputs, state, name=None):
        """Perform a decoding step.
        Args:
           time: scalar `int32` tensor
           inputs: A input tensors
           state: A state tensors and TensorArrays
           name (string): Name scope for any created operations
        Returns:
            outputs: An instance of AttentionDecoderOutput
            next_state: A state tensors and TensorArrays
            next_inputs: The tensor that should be used as input for the
                next step
            finished: A boolean tensor telling whether the sequence is
                complete, for each sequence in the batch
        """
        # このreuseいらない
        with tf.variable_scope("step", reuse=self.reuse):
            # Call LSTMCell
            cell_output_prev, cell_state_prev = self.rnn_cell(inputs, state)
            attention_weights_prev = self.attention_weights
            decoder_output, logits, attention_weights, context_vector = self.compute_output(
                cell_output_prev, attention_weights_prev)
            self.attention_weights = attention_weights

            sample_ids = self.helper.sample(time=time,
                                            outputs=logits,
                                            state=cell_state_prev)
            # TODO: Trainingのときlogitsの値はone-hotまたは一意のベクトルに変換されているか？

            outputs = AttentionDecoderOutput(logits=logits,
                                             predicted_ids=sample_ids,
                                             decoder_output=decoder_output,
                                             attention_weights=attention_weights,
                                             context_vector=context_vector)

            finished, next_inputs, next_state = self.helper.next_inputs(
                time=time,
                outputs=outputs,
                state=cell_state_prev,
                sample_ids=sample_ids)

            return outputs, next_state, next_inputs, finished
