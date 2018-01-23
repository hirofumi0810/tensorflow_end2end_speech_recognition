#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilitiy functions for beam search decoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def gather_tree_py(values, parents):
    """Gathers path through a tree backwards from the leave nodes. Used
    to reconstruct beams given their parents."""
    beam_length = values.shape[0]
    num_beams = values.shape[1]
    res = np.zeros_like(values)
    res[-1, :] = values[-1, :]
    for beam_id in range(num_beams):
        parent = parents[-1][beam_id]
        for level in reversed(range(beam_length - 1)):
            res[level, beam_id] = values[level][parent]
            parent = parents[level][parent]
    return np.array(res).astype(values.dtype)


def gather_tree(values, parents):
    """Tensor version of gather_tree_py."""
    res = tf.py_func(
        func=gather_tree_py, inp=[values, parents], Tout=values.dtype)
    res.set_shape(values.get_shape().as_list())
    return res


def mask_probs(probs, eos_token, finished):
    """Masks log probabilities such that finished beams allocate all
    probability mass to the EOS token and unfinished beams remain unchanged.
    Args:
        probs: Log probabiltiies of shape `[beam_width, vocab_size]`
        eos_token: An int32 id corresponding to the EOS token to allocate
            probability to
        finished: A boolean tensor of shape `[beam_width]` that specifies which
            elements in the beam are finished already
    Returns:
        A tensor of shape `[beam_width, vocab_size]`, where unfinished beams
        stay unchanged and finished beams are replaced with a tensor that has
        all probability on the EOS token.
    """
    vocab_size = tf.shape(probs)[1]
    finished_mask = tf.expand_dims(
        tf.to_float(1. - tf.to_float(finished)), axis=1)

    # These examples are not finished and we leave them
    non_finished_examples = finished_mask * probs

    # All finished examples are replaced with a vector that has all
    # probability on the EOS token
    finished_row = tf.one_hot(
        eos_token,
        vocab_size,
        dtype=tf.float32,
        on_value=0.,
        off_value=tf.float32.min)
    finished_examples = (1. - finished_mask) * finished_row

    return finished_examples + non_finished_examples


def normalize_score(log_probs, sequence_lengths, length_penalty_weight):
    """Normalizes scores for beam search hypotheses by the length.
    Args:
        log_probs: The log probabilities with shape
            `[beam_width, vocab_size]`.
        sequence_lengths: The sequence length of all hypotheses, a tensor
            of shape `[beam_size, vocab_size]`.
        length_penalty_weight: A float value, a scalar that weights the length
            penalty. Disabled with 0.0.
    Returns:
        score: The scores normalized by the length_penalty
    """
    # Calculate the length penality
    length_penality = tf.div(
        (5. + tf.to_float(sequence_lengths))**length_penalty_weight,
        (5. + 1.)**length_penalty_weight)
    # NOTE: See details in https://arxiv.org/abs/1609.08144.

    # Normalize log probabiltiies by the length penality
    if length_penalty_weight is None or length_penalty_weight == 1:
        score = log_probs
    else:
        score = log_probs / length_penality

    return score


def choose_top_k(scores_flat, beam_width):
    """Chooses the top-k beams as successors.
    Args:
        scores_flat:
        beam_width: int,
    Returns:
        next_beam_scores:
        pred_indices:
    """
    next_beam_scores, pred_indices = tf.nn.top_k(
        scores_flat, k=beam_width)
    return next_beam_scores, pred_indices


def nest_map(inputs, map_fn, name=None):
    """Applies a function to (possibly nested) tuple of tensors.
    """
    if nest.is_sequence(inputs):
        inputs_flat = nest.flatten(inputs)
        y_flat = [map_fn(_) for _ in inputs_flat]
        outputs = nest.pack_sequence_as(inputs, y_flat)
    else:
        outputs = map_fn(inputs)
    if name:
        outputs = tf.identity(outputs, name=name)
    return outputs
