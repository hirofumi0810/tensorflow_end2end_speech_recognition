#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Vanilla CTC model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.ctc.base import CTCBase
from models.encoders.load_encoder import load


class CTC(CTCBase):
    """Vanilla CTC model.
    Args:
        encoder_type (string): The type of an encoder
            blstm: bidirectional LSTM
            lstm: unidirectional LSTM
            bgru: bidirectional GRU
            gru: unidirectional GRU
            vgg_blstm: VGG + bidirectional LSTM
            vgg_lstm: VGG + unidirectional LSTM
            cnn: CNN
        input_size (int): the dimensions of input vectors
        num_units (int): the number of units in each layer
        num_layers (int): the number of layers
        num_classes (int): the number of classes of target labels
            (except for a blank label)
        lstm_impl (string, optional): a base implementation of LSTM. This is
            not used for GRU models.
                BasicLSTMCell: tf.contrib.rnn.BasicLSTMCell (no peephole)
                LSTMCell: tf.contrib.rnn.LSTMCell
                LSTMBlockCell: tf.contrib.rnn.LSTMBlockCell
                LSTMBlockFusedCell: under implementation
            Choose the background implementation of tensorflow.
            Default is LSTMBlockCell (the fastest).
        use_peephole (bool, optional): if True, use peephole connection. This
            is not used for GRU models.
        splice (int, optional): the number of frames to splice. This is used
            when using CNN-like encoder. Default is 1 frame.
        parameter_init (float, optional): the range of uniform distribution to
            initialize weight parameters (>= 0)
        clip_grad (float, optional): the range of gradient clipping (> 0)
        clip_activation (float, optional): the range of activation clipping
            (> 0). This is not used for GRU models.
        num_proj (int, optional): the number of nodes in the projection layer.
            This is not used for GRU models.
        weight_decay (float, optional): a parameter for weight decay
        bottleneck_dim (int, optional): the dimensions of the bottleneck layer
    """

    def __init__(self,
                 encoder_type,
                 input_size,
                 num_units,
                 num_layers,
                 num_classes,
                 lstm_impl='LSTMBlockCell',
                 use_peephole=True,
                 splice=1,
                 parameter_init=0.1,
                 clip_grad=None,
                 clip_activation=None,
                 num_proj=None,
                 weight_decay=0.0,
                 bottleneck_dim=None):

        super(CTC, self).__init__(
            input_size, splice, num_classes, clip_grad, weight_decay)
        self.name = encoder_type + '_ctc'

        if encoder_type in ['blstm', 'lstm']:
            self.encoder = load(encoder_type)(num_units=num_units,
                                              num_layers=num_layers,
                                              num_classes=num_classes + 1,
                                              lstm_impl=lstm_impl,
                                              use_peephole=use_peephole,
                                              parameter_init=parameter_init,
                                              clip_activation=clip_activation,
                                              num_proj=num_proj,
                                              bottleneck_dim=bottleneck_dim)

        elif encoder_type in ['vgg_blstm', 'vgg_lstm']:
            self.encoder = load(encoder_type)(input_size=input_size,
                                              splice=splice,
                                              num_units=num_units,
                                              num_layers=num_layers,
                                              num_classes=num_classes + 1,
                                              lstm_impl=lstm_impl,
                                              use_peephole=use_peephole,
                                              parameter_init=parameter_init,
                                              clip_activation=clip_activation,
                                              num_proj=num_proj,
                                              bottleneck_dim=bottleneck_dim)

        elif encoder_type in ['bgru', 'gru']:
            self.encoder = load(encoder_type)(num_units=num_units,
                                              num_layers=num_layers,
                                              num_classes=num_classes + 1,
                                              parameter_init=parameter_init,
                                              bottleneck_dim=bottleneck_dim)

        elif encoder_type in ['vgg_wang', 'resnet_wang', 'cnn']:
            self.encoder = load(encoder_type)(input_size=input_size,
                                              splice=splice,
                                              num_classes=num_classes + 1,
                                              parameter_init=parameter_init)

        else:
            raise NotImplementedError
