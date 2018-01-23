#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Classes for namedtuple."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple


class FinalBeamDecoderOutput(namedtuple(
        "FinalBeamDecoderOutput",
        [
            "predicted_ids",
            "beam_search_output"
        ])):
    """Final outputs returned by the beam search after all decoding is finished.
    Args:
        predicted_ids: The final prediction. A tensor of shape
            `[time, 1, beam_width]???`.
        beam_search_output: An instance of `BeamSearchDecoderOutput` that
            describes the state of the beam search.
    """
    pass


class BeamSearchDecoderOutput(namedtuple(
        "BeamSearchDecoderOutput",
        [
            "logits",
            "predicted_ids",
            "log_probs",
            "scores",
            "beam_parent_ids",
            "original_outputs"
        ])):
    """Structure for the output of a beam search decoder. This class is used
    to define the output at each step as well as the final output of the
    decoder. If used as the final output, a time dimension `time` is inserted
    after the beam_size dimension.
    Args:
        logits: Logits at the current time step of shape
            `[beam_width, num_classes]`
        predicted_ids: Chosen softmax predictions at the current time step.
            An int32 tensor of shape `[beam_width]`.
        log_probs: Total log probabilities of all beams at the current time
            step. A float32 tensor of shaep `[beam_width]`.
        scores: Total scores of all beams at the current time step. This
            differs from log probabilities in that the score may add additional
            processing such as length normalization. A float32 tensor of shape
            `[beam_width]`.
        beam_parent_ids: The indices of the beams that are being continued.
            An int32 tensor of shape `[beam_width]`.
        original_outputs:
    """
    pass


class BeamSearchDecoderState(namedtuple(
        "BeamSearchDecoderState",
        [
            "log_probs",
            "finished",
            "lengths"
        ])):
    """State for a single step of beam search.
    Args:
        log_probs: The current log probabilities of all beams
        finished: A boolean vector that specifies which beams are finished
        lengths: Lengths of all beams
    """
    pass


class BeamSearchStepOutput(namedtuple(
        "BeamSearchStepOutput",
        [
            "scores",
            "predicted_ids",
            "beam_parent_ids"
        ])):
    """Outputs for a single step of beam search.
    Args:
        scores: Score for each beam, a float32 vector
        predicted_ids: predictions for this step, an int32 vector
        beam_parent_ids: an int32 vector containing the beam indices of the
            continued beams from the previous step
    """
    pass
