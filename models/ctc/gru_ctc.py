#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""(Bidirectional) GRU-CTC model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.ctc.ctc_base import CTCBase
from models.encoders.bgru_encoder import BGRU_Encoder
from models.encoders.gru_encoder import GRU_Encoder


class GRU_CTC(CTCBase):
    """(Bidirectional) GRU-CTC model.
    Args:
        input_size (int): the dimensions of input vectors
        num_units (int): the number of units in each layer
        num_layers (int): the number of layers
        num_classes (int): the number of classes of target labels
            (except for a blank label)
        bidirectional (bool): if True, create a bidirectional model
        lstm_impl: <not used>
        use_peephole: <not used>
        splice (int, optional): frames to splice. Default is 1 frame.
        parameter_init (float, optional): Range of uniform distribution to
            initialize weight parameters
        clip_grad (float, optional): Range of gradient clipping (> 0)
        clip_activation: <not used>
        num_proj: <not used>
        weight_decay (float, optional): Regularization parameter for weight decay
        bottleneck_dim (int, optional): the dimensions of the bottleneck layer
    """

    def __init__(self,
                 input_size,
                 num_units,
                 num_layers,
                 num_classes,
                 bidirectional,
                 lstm_impl=None,  # <not used>
                 use_peephole=False,  # <not used>
                 splice=1,
                 parameter_init=0.1,
                 clip_grad=None,
                 clip_activation=None,  # <not used>
                 num_proj=None,  # <not used>
                 weight_decay=0.0,
                 bottleneck_dim=None):

        super(GRU_CTC, self).__init__(
            input_size, splice, num_classes, clip_grad, weight_decay)

        if bidirectional:
            self.name = 'bgru_ctc'
            self.encoder = BGRU_Encoder(num_units,
                                        num_layers,
                                        num_classes + 1,
                                        parameter_init,
                                        bottleneck_dim)
        else:
            self.name = 'gru_ctc'
            self.encoder = GRU_Encoder(num_units,
                                       num_layers,
                                       num_classes + 1,
                                       parameter_init,
                                       bottleneck_dim)
