#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Select & load encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .lstm_encoder import LSTMEncoder
from .blstm_encoder import BLSTMEncoder
from .gru_encoder import GRUEncoder
from .bgru_encoder import BGRUEncoder

Encoder = {
    "lstm_encoder": LSTMEncoder,
    "blstm_encoder": BLSTMEncoder,
    "gru_encoder": GRUEncoder,
    "bgru_encoder": BGRUEncoder
}


def load(model_type):
    """Select & load model.
    Args:
        model_type: string, name of the encoder in the key of Encoder
    Returns:
        model: class object
    """
    if model_type not in Encoder:
        raise ValueError(
            "model_type should be one of [%s], you provided %s." %
            (", ".join(Encoder), model_type))

    return Encoder[model_type]
