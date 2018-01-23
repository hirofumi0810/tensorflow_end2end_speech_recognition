#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""A decoder that uses beam search. Can only be used for inference, not
training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.util import nest

from models.attention.decoders.dynamic_decoder import dynamic_decode
from models.attention.decoders.attention_decoder import RNNDecoder
from models.attention.decoders.beam_search.util import gather_tree
from models.attention.decoders.beam_search.util import mask_probs
from models.attention.decoders.beam_search.util import normalize_score
from models.attention.decoders.beam_search.namedtuple import FinalBeamDecoderOutput
from models.attention.decoders.beam_search.namedtuple import BeamSearchDecoderOutput
from models.attention.decoders.beam_search.namedtuple import BeamSearchDecoderState
from models.attention.decoders.beam_search.namedtuple import BeamSearchStepOutput


class BeamSearchDecoder(RNNDecoder):
    """The BeamSearchDecoder wraps another decoder to perform beam search
    instead of greedy selection. This decoder must be used with batch size of
    1, which will result in an effective batch size of `beam_width`.
    Args:
        decoder: An instance of `RNNDecoder` to be used with beam search
        beam_width: int, the number of beams to use
        vocab_size: int, the number of classses
        eos_index: int, the id of the EOS token, used to mark beams as "done"
        length_penalty_weight: A float value, weight for the length penalty
            factor. 0.0　disables the penalty.
        choose_successors_fn: A function used to choose beam successors based
            on their scores. Maps from
            `(scores, config)` => `(chosen scores, chosen_ids)`
    """

    def __init__(self, decoder, beam_width, vocab_size, eos_index,
                 length_penalty_weight, choose_successors_fn,
                 name='beam_search_decoder'):
        super(BeamSearchDecoder, self).__init__(
            decoder.cell,
            decoder.parameter_init,
            decoder.max_decode_length,
            decoder.num_classes,
            decoder.attention_encoder_states,
            decoder.attention_values,
            decoder.attention_values_length,
            decoder.attention_layer,
            decoder.time_major,
            name)

        self.decoder = decoder
        self.beam_width = beam_width
        self.vocab_size = vocab_size
        self.eos_index = eos_index
        self.length_penalty_weight = length_penalty_weight
        self.choose_successors_fn = choose_successors_fn

    def __call__(self, *args, **kwargs):
        with tf.variable_scope(self.decoder.name, reuse=True):
            return self._build(*args, **kwargs)

    @property
    def output_size(self):
        return BeamSearchDecoderOutput(
            logits=self.decoder.num_classes,
            predicted_ids=tf.TensorShape([]),
            log_probs=tf.TensorShape([]),
            scores=tf.TensorShape([]),
            beam_parent_ids=tf.TensorShape([]),
            original_outputs=self.decoder.output_size)

    @property
    def output_dtype(self):
        return BeamSearchDecoderOutput(
            logits=tf.float32,
            predicted_ids=tf.int32,
            log_probs=tf.float32,
            scores=tf.float32,
            beam_parent_ids=tf.int32,
            original_outputs=self.decoder.output_dtype)

    @property
    def batch_size(self):
        # return tf.shape(nest.flatten([self.initial_state])[0])[0]
        return self.beam_width

    def _build(self, initial_state, helper, mode):
        """
        Args:
            initial_state: A tensor or tuple of tensors used as the initial
                cell state. Set to the final state of the encoder by default.
            helper: An instance of `tf.contrib.seq2seq.Helper` to assist
                decoding
        Returns:
            An instance of `RNNDecoder`
        """
        print('===== beam search decoder build =====')

        # Tile initial state
        initial_state = nest.map_structure(
            lambda x: tf.tile(x, [self.batch_size, 1]), initial_state)
        self.decoder._setup(initial_state, helper)

        self.reuse = True
        maximum_iterations = self.max_decode_length

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

    def finalize(self, outputs, final_state, final_seq_len):
        """
        Args:
            outputs: An instance of `BeamSearchDecoderOutput`
            final_state:
            final_seq_len:
        Returns:
            outputs:
            final_state:
        """
        print('===== finalize (beam search) =====')
        # Gather according to beam search result
        predicted_ids = gather_tree(outputs.predicted_ids,
                                    outputs.beam_parent_ids)

        # We're using a batch size of 1, so we add an extra dimension to
        # convert tensors to [1, beam_width, ...] shape. This way Tensorflow
        # doesn't confuse batch_size with beam_width
        outputs = nest.map_structure(
            lambda x: tf.expand_dims(x, axis=1), outputs)

        outputs = FinalBeamDecoderOutput(
            predicted_ids=tf.expand_dims(predicted_ids, axis=1),
            beam_search_output=outputs)

        return outputs, final_state

    def initialize(self, name=None):
        """Initialize the decoder.
        Args:
            name: Name scope for any created operations
        Returns:
            finished:
            first_inputs:
            initial_state:
        """
        print('===== initialize (beam search) =====')
        # Create inputs for the first time step
        finished, first_inputs, initial_state = self.decoder.initialize()

        # Create initial beam state
        beam_state = BeamSearchDecoderState(
            log_probs=tf.zeros([self.beam_width]),
            finished=tf.zeros([self.beam_width], dtype=tf.bool),
            lengths=tf.zeros([self.beam_width], dtype=tf.int32))

        return finished, first_inputs, (initial_state, beam_state)

    def step(self, time, inputs, state, name=None):
        """Perform a decoding step.
        Args:
            time: scalar `int32` tensor
            inputs: A input tensors
            state: A state tensors and TensorArrays
            name: Name scope for any created operations
        Returns:
            outputs: An instance of `BeamSearchDecoderOutput`
            next_state: A state tensors and TensorArrays
            next_inputs: The tensor that should be used as input for the
                next step
            finished: A boolean tensor telling whether the sequence is
                complete, for each sequence in the batch
        """
        print('===== step (beam search) =====')
        decoder_state, beam_state = state

        # Call the original decoder
        decoder_output, decoder_state, _, _ = self.decoder.step(time, inputs,
                                                                decoder_state)

        # Perform a step of beam search
        beam_search_output, beam_state = beam_search_step(
            time=time,
            logits=decoder_output.logits,
            beam_state=beam_state,
            beam_width=self.beam_width,
            vocab_size=self.vocab_size,
            eos_index=self.eos_index,
            length_penalty_weight=self.length_penalty_weight,
            choose_successors_fn=self.choose_successors_fn)

        # Shuffle everything according to beam search result
        decoder_state = nest.map_structure(
            lambda x: tf.gather(x, beam_search_output.beam_parent_ids),
            decoder_state)
        decoder_output = nest.map_structure(
            lambda x: tf.gather(x, beam_search_output.beam_parent_ids),
            decoder_output)

        next_state = (decoder_state, beam_state)

        outputs = BeamSearchDecoderOutput(
            logits=tf.zeros([self.beam_width, self.vocab_size]),
            predicted_ids=beam_search_output.predicted_ids,
            log_probs=beam_state.log_probs,
            scores=beam_search_output.scores,
            beam_parent_ids=beam_search_output.beam_parent_ids,
            original_outputs=decoder_output)

        finished, next_inputs, next_state = self.decoder.helper.next_inputs(
            time=time,
            outputs=decoder_output,
            state=next_state,
            sample_ids=beam_search_output.predicted_ids)
        next_inputs.set_shape([self.batch_size, None])

        return outputs, next_state, next_inputs, finished


def beam_search_step(time, logits, beam_state, beam_width, vocab_size,
                     eos_index, length_penalty_weight, choose_successors_fn):
    """Performs a single step of Beam Search Decoding.
    Args:
        time: Beam search time step, should start at 0. At time 0 we assume
            that all beams are equal and consider only the first beam for
            continuations
        logits: Logits at the current time step. A tensor of shape
            `[beam_width, vocab_size]`
        beam_state: Current state of the beam search. An instance of
            `BeamState??`
        beam_width: int, the number of beams to use
        vocab_size: int, the number of classses
        eos_index: int, the id of the EOS token, used to mark beams as "done"
        length_penalty_weight: A float value, weight for the length penalty
            factor. 0.0 disables the penalty.
        choose_successors_fn: A function used to choose beam successors based
            on their scores. Maps from
            `(scores, config)` => `(chosen scores, chosen_ids)`
    Returns:
        output: An instance of `BeamSearchStepOutput`
        next_state: An instance of `BeamSearchDecoderState`
        A new beam state
    """
    # Calculate the current lengths of the predictions
    prediction_lengths = beam_state.lengths
    previously_finished = beam_state.finished

    # Calculate the total log probs for the new hypotheses
    # Final Shape: `[beam_width, vocab_size]`
    probs = tf.nn.log_softmax(logits)
    probs = mask_probs(probs, eos_index, previously_finished)
    total_probs = tf.expand_dims(beam_state.log_probs, axis=1) + probs

    # Calculate the continuation lengths
    # We add 1 to all continuations that are not EOS and were not
    # finished previously
    lengths_to_add = tf.one_hot([eos_index] * beam_width,
                                vocab_size,
                                on_value=0,
                                off_value=1)
    add_mask = (1 - tf.to_int32(previously_finished))
    lengths_to_add = tf.expand_dims(add_mask, axis=1) * lengths_to_add
    new_prediction_lengths = tf.expand_dims(
        prediction_lengths, axis=1) + lengths_to_add

    # Calculate the scores for each beam
    scores = normalize_score(
        log_probs=total_probs,
        sequence_lengths=new_prediction_lengths,
        length_penalty_weight=length_penalty_weight)
    scores_flat = tf.reshape(scores, shape=[-1])

    # During the first time step, we only consider the initial beam
    scores_flat = tf.cond(tf.convert_to_tensor(time) > 0,
                          true_fn=lambda: scores_flat,  # otherwise
                          false_fn=lambda: scores[0])   # first time step

    # Pick the next beams according to the specified successors function
    next_beam_scores, word_indices = choose_successors_fn(
        scores_flat,
        beam_width)
    next_beam_scores.set_shape([beam_width])
    word_indices.set_shape([beam_width])

    # Pick out the probs, beam_ids, and states according to the chosen
    # predictions
    total_probs_flat = tf.reshape(total_probs, shape=[-1],
                                  name="total_probs_flat")
    next_beam_probs = tf.gather(total_probs_flat, word_indices)
    next_beam_probs.set_shape([beam_width])
    next_word_ids = tf.mod(word_indices, vocab_size)
    next_beam_ids = tf.div(word_indices, vocab_size)

    # Append new ids to current predictions
    next_finished = tf.logical_or(
        tf.gather(beam_state.finished, next_beam_ids),
        tf.equal(next_word_ids, eos_index))

    # Calculate the length of the next predictions.
    # 1. Finished beams remain unchanged
    # 2. Beams that are now finished (EOS predicted) remain unchanged
    # 3. Beams that are not yet finished have their length increased by 1
    lengths_to_add = tf.to_int32(tf.not_equal(next_word_ids, eos_index))
    lengths_to_add = (1 - tf.to_int32(next_finished)) * lengths_to_add
    next_prediction_len = tf.gather(beam_state.lengths, next_beam_ids)
    next_prediction_len += lengths_to_add

    next_state = BeamSearchDecoderState(
        log_probs=next_beam_probs,
        lengths=next_prediction_len,
        finished=next_finished)

    output = BeamSearchStepOutput(
        scores=next_beam_scores,
        predicted_ids=next_word_ids,
        beam_parent_ids=next_beam_ids)

    return output, next_state
