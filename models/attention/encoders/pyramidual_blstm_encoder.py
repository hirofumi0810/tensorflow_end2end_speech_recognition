#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Pyramidual Bidirectional LSTM Encoder class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import tensorflow as tf

# template
EncoderOutput = namedtuple(
    "EncoderOutput",
    [
        "outputs",
        "final_state",
        "attention_values",
        "attention_values_length"
    ])


class PyramidualBLSTMEncoder(object):
    """Pyramidual Bidirectional LSTM Encoder.
    Args:
        num_cell:
        num_layer:
        keep_prob_input:
        keep_prob_hidden:
        parameter_init:
        clip_activation:
        num_proj:
    """

    def __init__(self,
                 num_cell,
                 num_layer,
                 keep_prob_input=1.0,
                 keep_prob_hidden=1.0,
                 parameter_init=0.1,
                 clip_activation=50,
                 num_proj=None,
                 name='pblstm_encoder'):

        self.num_cell = num_cell
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

    def _build(self, inputs, seq_len):
        """Construct Bidirectional GRU encoder.
        Args:
            inputs:
            seq_len:
        Returns:
            EncoderOutput: A tuple of
                `(outputs, final_state,
                        attention_values, attention_values_length)`
                outputs:
                final_state:
                attention_values:
                attention_values_length:
        """
        self.inputs = inputs
        self.seq_len = seq_len
        
        raise NotImplementedError()
