#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Select & load encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.attention.encoders.lstm_encoder import LSTMEncoder
from models.attention.encoders.blstm_encoder import BLSTMEncoder
from models.attention.encoders.gru_encoder import GRUEncoder
from models.attention.encoders.bgru_encoder import BGRUEncoder
from models.attention.encoders.pyramidal_blstm_encoder import PyramidalBLSTMEncoder

Encoder = {
    "lstm_encoder": LSTMEncoder,
    "blstm_encoder": BLSTMEncoder,
    "gru_encoder": GRUEncoder,
    "bgru_encoder": BGRUEncoder,
    "pblstm_encoder": PyramidalBLSTMEncoder
}


def load(model_type):
    """Select & load model.
    Args:
        model_type: string, name of the encoder in the key of Encoder
    Returns:
        model: An instance of RNN encoder
    """
    if model_type not in Encoder:
        raise ValueError(
            "model_type should be one of [%s], you provided %s." %
            (", ".join(Encoder), model_type))

    return Encoder[model_type]
