#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Bidirectional LSTM-CTC model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from models.ctc.ctc_base import ctcBase
from models.encoders.blstm_encoder import BLSTM_Encoder


class BLSTM_CTC(ctcBase):
    """Bidirectional LSTM-CTC model.
    Args:
        input_size: int, the dimensions of input vectors
        num_unit: int, the number of units in each layer
        num_layer: int, the number of layers
        num_classes: int, the number of classes of target labels
            (except for a blank label)
        lstm_impl: string, BasicLSTMCell or LSTMCell or or LSTMBlockCell or
            LSTMBlockFusedCell.
            Choose the background implementation of tensorflow.
            Default is LSTMBlockCell (the fastest implementation).
        use_peephole: bool, if True, use peephole
        splice: int, frames to splice. Default is 1 frame.
        parameter_init: A float value. Range of uniform distribution to
            initialize weight parameters
        clip_grad: A float value. Range of gradient clipping (> 0)
        clip_activation: A float value. Range of activation clipping (> 0)
        num_proj: int, the number of nodes in recurrent projection layer
        weight_decay: A float value. Regularization parameter for weight decay
        bottleneck_dim: int, the dimensions of the bottleneck layer
        name: string, the name of the CTC model
    """

    def __init__(self,
                 input_size,
                 num_unit,
                 num_layer,
                 num_classes,
                 lstm_impl='LSTMBlockCell',
                 use_peephole=True,
                 splice=1,
                 parameter_init=0.1,
                 clip_grad=None,
                 clip_activation=None,
                 num_proj=None,
                 weight_decay=0.0,
                 bottleneck_dim=None,
                 name='blstm_ctc'):

        ctcBase.__init__(self, input_size, splice, num_classes,
                         clip_grad, weight_decay)

        self.name = name
        self.encoder = BLSTM_Encoder(num_unit,
                                     num_layer,
                                     num_classes,
                                     lstm_impl,
                                     use_peephole,
                                     parameter_init,
                                     clip_activation,
                                     num_proj,
                                     bottleneck_dim)

    def __call__(self, inputs, inputs_seq_len, keep_prob_input,
                 keep_prob_hidden, keep_prob_output):
        """Construct model graph.
        Args:
            inputs: A tensor of size `[B, T, input_size]`
            inputs_seq_len: A tensor of size `[B]`
            keep_prob_input: A float value. A probability to keep nodes in
                the input-hidden connection
            keep_prob_hidden: A float value. A probability to keep nodes in
                the hidden-hidden connection
            keep_prob_output: A float value. A probability to keep nodes in
                the hidden-output connection
        Returns:
            logits: A tensor of size `[T, B, num_classes]`
        """
        logits = self.encoder(inputs, inputs_seq_len, keep_prob_input,
                              keep_prob_hidden, keep_prob_output)

        return logits
