#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class of encoders."""

from collections import namedtuple
import tensorflow as tf


class EncoderOutput(
    namedtuple("EncoderOutput",
               ["outputs",
                "final_state",
                "attention_values",
                "attention_values_length"])):
    pass


class EncoderBase(object):
    """Base class of the encoder.
    Args:
        num_units:
        num_layer:
        keep_prob_input:
        keep_prob_hidden:
        parameter_init:
        clip_activation:
        num_proj:
    """

    def __init__(self,
                 num_units,
                 num_layer,
                 keep_prob_input,
                 keep_prob_hidden,
                 parameter_init,
                 clip_activation,
                 num_proj,
                 name=None):

        self.num_units = num_units
        self.num_layer = num_layer
        self.keep_prob_input = keep_prob_input
        self.keep_prob_hidden = keep_prob_hidden
        self.parameter_init = parameter_init
        self.clip_activation = clip_activation
        self.num_proj = num_proj
        self.name = name

    def __call__(self, *args, **kwargs):
        # TODO: variable_scope
        with tf.name_scope('Encoder'):
            return self._build(*args, **kwargs)

    def _build(self, inputs, inputs_seq_len):
        raise NotImplementedError
