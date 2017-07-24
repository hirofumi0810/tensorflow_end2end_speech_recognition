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
        num_unit: int, the number of units in each layer
        num_layer: int, the number of layers
        parameter_init: A float value. Range of uniform distribution to
            initialize weight parameters
        clip_activation: A float value. Range of activation clipping (> 0)
        num_proj: int, the number of nodes in recurrent projection layer
    """

    def __init__(self,
                 num_unit,
                 num_layer,
                 parameter_init,
                 clip_activation,
                 num_proj,
                 name=None):

        self.num_unit = num_unit
        self.num_layer = num_layer
        self.parameter_init = parameter_init
        self.clip_activation = clip_activation
        self.num_proj = num_proj
        self.name = name

    def __call__(self, *args, **kwargs):
        with tf.variable_scope(self.name):
            return self._build(*args, **kwargs)

    def _build(self, inputs, inputs_seq_len):
        raise NotImplementedError
