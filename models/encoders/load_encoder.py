#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Select & load encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.encoders.core.blstm import BLSTMEncoder
from models.encoders.core.lstm import LSTMEncoder
from models.encoders.core.gru import GRUEncoder, BGRUEncoder
from models.encoders.core.cnn_zhang import CNNEncoder
from models.encoders.core.vgg_blstm import VGGBLSTMEncoder
from models.encoders.core.vgg_lstm import VGGLSTMEncoder
from models.encoders.core.vgg_wang import VGGEncoder
# from models.encoders.core.resnet_wang import ResNetEncoder
from models.encoders.core.multitask_blstm import MultitaskBLSTMEncoder
from models.encoders.core.multitask_lstm import MultitaskLSTMEncoder
from models.encoders.core.pyramidal_blstm import PyramidBLSTMEncoder

ENCODERS = {
    "blstm": BLSTMEncoder,
    "lstm": LSTMEncoder,
    "bgru": BGRUEncoder,
    "gru": GRUEncoder,
    "vgg_blstm": VGGBLSTMEncoder,
    "vgg_lstm": VGGLSTMEncoder,
    "cnn_zhang": CNNEncoder,
    "vgg_wang": VGGEncoder,
    # "resnet_wang": ResNetEncoder,
    "multitask_blstm": MultitaskBLSTMEncoder,
    "multitask_lstm": MultitaskLSTMEncoder,
    "pyramid_blstm": PyramidBLSTMEncoder,
}


def load(encoder_type):
    """Select & load encoder.
    Args:
        encoder_type (string): name of the ctc model in the key of ENCODERS
    Returns:
        An instance of the encoder
    """
    if encoder_type not in ENCODERS.keys():
        raise ValueError(
            "encoder_type should be one of [%s], you provided %s." %
            (", ".join(ENCODERS), encoder_type))
    return ENCODERS[encoder_type]
