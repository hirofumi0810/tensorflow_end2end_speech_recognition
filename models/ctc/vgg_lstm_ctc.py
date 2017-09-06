#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""VGG + (bidirectional) LSTM-CTC model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.ctc.ctc_base import CTCBase
from models.encoders.vgg_blstm_encoder import VGG_BLSTM_Encoder
# from models.encoders.vgg_lstm_encoder import VGG_LSTM_Encoder


class VGG_LSTM_CTC(CTCBase):
    """VGG + (bidirectional) LSTM-CTC model.
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
        use_peephole (bool, optional): if True, use peephole
        splice (int, optional): frames to splice. Default is 1 frame.
        parameter_init (float, optional): Range of uniform distribution to
            initialize weight parameters
        clip_grad (float, optional): Range of gradient clipping (> 0)
        clip_activation (float, optional): Range of activation clipping (> 0)
        num_proj (int, optional): the number of nodes in recurrent projection layer
        weight_decay (float, optional): Regularization parameter for weight decay
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
                 splice=11,
                 parameter_init=0.1,
                 clip_grad=None,
                 clip_activation=None,
                 num_proj=None,
                 weight_decay=0.0,
                 bottleneck_dim=None):

        super(VGG_LSTM_CTC, self).__init__(
            input_size, splice, num_classes, clip_grad, weight_decay)

        if bidirectional:
            self.name = 'vgg_blstm_ctc'
            self.encoder = VGG_BLSTM_Encoder(input_size,
                                             num_units,
                                             num_layers,
                                             num_classes + 1,
                                             lstm_impl,
                                             use_peephole,
                                             splice,
                                             parameter_init,
                                             clip_activation,
                                             num_proj,
                                             bottleneck_dim)

        else:
            self.name = 'vgg_lstm_ctc'
            raise NotImplementedError
            # self.encoder = VGG_LSTM_Encoder(input_size,
            #                                 num_units,
            #                                 num_layers,
            #                                 num_classes + 1,
            #                                 lstm_impl,
            #                                 use_peephole,
            #                                 splice,
            #                                 parameter_init,
            #                                 clip_activation,
            #                                 num_proj,
            #                                 bottleneck_dim)
