#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Pyramidal Bidirectional LSTM Encoder class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .encoder_base import EncoderOutput, EncoderBase


class PyramidalBLSTMEncoder(EncoderBase):
    """Pyramidal Bidirectional LSTM Encoder.
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

        EncoderBase.__init__(self, num_cell, num_layer, keep_prob_input,
                             keep_prob_hidden, parameter_init, clip_activation,
                             num_proj, name)

    def _build(self, inputs, seq_len):
        """Construct Pyramidal Bidirectional LSTM encoder.
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

        raise NotImplementedError
