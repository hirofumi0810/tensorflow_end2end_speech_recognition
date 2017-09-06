#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""CNN-CTC model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.ctc.ctc_base import CTCBase
from models.encoders.cnn_encoder import CNN_Encoder


class CNN_CTC(CTCBase):
    """CNN-CTC model.
       This implementaion is based on
           https://arxiv.org/abs/1701.02720.
               Zhang, Ying, et al.
               "Towards end-to-end speech recognition with deep convolutional
                neural networks."
               arXiv preprint arXiv:1701.02720 (2017).
    Args:
        input_size (int): the dimensions of input vectors
        num_units: <not used>
        num_layers: <not used>
        num_classes (int): the number of classes of target labels
            (except for a blank label)
        bidirectional: <not used>
        lstm_impl: <not used>
        use_peephole: <not used>
        splice (int, optional): frames to splice. Default is 1 frame.
        parameter_init (float, optional): Range of uniform distribution to
            initialize weight parameters
        clip_grad (float, optional): Range of gradient clipping (> 0)
        clip_activation (float, optional): Range of activation clipping (> 0)
        num_proj: <not used>
        weight_decay (float, optional): Regularization parameter for weight decay
        bottleneck_dim: <not used>
    """

    def __init__(self,
                 input_size,
                 num_classes,
                 num_units=0,  # <not used>
                 num_layers=0,  # <not used>
                 bidirectional=False,  # <not used>
                 lstm_impl=None,  # <not used>
                 use_peephole=False,  # <not used>
                 splice=11,
                 parameter_init=0.1,
                 clip_grad=None,
                 clip_activation=None,
                 num_proj=None,  # <not used>
                 weight_decay=0.0,
                 bottleneck_dim=None):  # <not used>

        super(CNN_CTC, self).__init__(
            input_size, splice, num_classes, clip_grad, weight_decay)

        self.name = 'cnn_ctc'
        self.encoder = CNN_Encoder(input_size,
                                   splice,
                                   num_classes + 1,
                                   parameter_init)
