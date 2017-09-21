#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Select & load encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.encoders.core.blstm import BLSTM_Encoder
from models.encoders.core.lstm import LSTM_Encoder
from models.encoders.core.bgru import BGRU_Encoder
from models.encoders.core.gru import GRU_Encoder
from models.encoders.core.cnn import CNN_Encoder
from models.encoders.core.vgg_blstm import VGG_BLSTM_Encoder
from models.encoders.core.vgg_lstm import VGG_LSTM_Encoder
from models.encoders.core.multitask_blstm import Multitask_BLSTM_Encoder
from models.encoders.core.multitask_lstm import Multitask_LSTM_Encoder

ENCODERS = {
    "blstm": BLSTM_Encoder,
    "lstm": LSTM_Encoder,
    "bgru": BGRU_Encoder,
    "gru": GRU_Encoder,
    "cnn": CNN_Encoder,
    "vgg_blstm": VGG_BLSTM_Encoder,
    "vgg_lstm": VGG_LSTM_Encoder,
    "multitask_blstm": Multitask_BLSTM_Encoder,
    "multitask_lstm": Multitask_LSTM_Encoder,
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
