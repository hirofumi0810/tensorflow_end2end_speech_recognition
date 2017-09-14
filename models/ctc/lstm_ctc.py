#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""(Bidirectional) LSTM-CTC model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.ctc.base import CTCBase
from models.encoders.blstm_encoder import BLSTM_Encoder
from models.encoders.lstm_encoder import LSTM_Encoder


class LSTM_CTC(CTCBase):
    """(Bidirectional) LSTM-CTC model.
    Args:
        input_size (int): the dimensions of input vectors
        num_units (int): the number of units in each layer
        num_layers (int): the number of layers
        num_classes (int): the number of classes of target labels
            (except for a blank label)
        bidirectional (bool): if True, create a bidirectional model
        lstm_impl (string, optional): BasicLSTMCell or LSTMCell or or
            LSTMBlockCell or LSTMBlockFusedCell.
            Choose the background implementation of tensorflow.
            Default is LSTMBlockCell (the fastest implementation).
        use_peephole (bool, optional): if True, use peephole connection
        splice (int, optional): the number of frames to splice.
            Default is 1 frame.
        parameter_init (float, optional): the range of uniform distribution to
            initialize weight parameters (>= 0)
        clip_grad (float): the range of gradient clipping (> 0)
        clip_activation (float, optional): the range of activation clipping (> 0)
        num_proj (int, optional): the number of nodes in the projection layer
        weight_decay (float, optional): a parameter for weight decay
        bottleneck_dim (int, optional): the dimensions of the bottleneck layer
    """

    def __init__(self,
                 input_size,
                 num_units,
                 num_layers,
                 num_classes,
                 bidirectional,
                 lstm_impl='LSTMBlockCell',
                 use_peephole=True,
                 splice=1,
                 parameter_init=0.1,
                 clip_grad=None,
                 clip_activation=None,
                 num_proj=None,
                 weight_decay=0.0,
                 bottleneck_dim=None):

        super(LSTM_CTC, self).__init__(
            input_size, splice, num_classes, clip_grad, weight_decay)

        if bidirectional:
            self.name = 'blstm_ctc'
            self.encoder = BLSTM_Encoder(num_units,
                                         num_layers,
                                         num_classes + 1,
                                         lstm_impl,
                                         use_peephole,
                                         parameter_init,
                                         clip_activation,
                                         num_proj,
                                         bottleneck_dim)
        else:
            self.name = 'lstm_ctc'
            self.encoder = LSTM_Encoder(num_units,
                                        num_layers,
                                        num_classes + 1,
                                        lstm_impl,
                                        use_peephole,
                                        parameter_init,
                                        clip_activation,
                                        num_proj,
                                        bottleneck_dim)
