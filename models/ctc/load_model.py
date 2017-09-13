#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Select & load CTC model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.ctc.lstm_ctc import LSTM_CTC
from models.ctc.gru_ctc import GRU_CTC
from models.ctc.cnn_ctc import CNN_CTC
from models.ctc.vgg_lstm_ctc import VGG_LSTM_CTC
from models.ctc.multitask_lstm_ctc import Multitask_LSTM_CTC


CTC = {
    "lstm_ctc": LSTM_CTC,
    "gru_ctc": GRU_CTC,
    "cnn_ctc": CNN_CTC,
    "vgg_lstm_ctc": VGG_LSTM_CTC,
    "multitask_lstm_ctc": Multitask_LSTM_CTC
}


def load(model_type):
    """Select & load model.
    Args:
        model_type (string): name of the ctc model in the key of CTC
    Returns:
        model: class object
    """
    if model_type not in CTC:
        raise ValueError(
            "model_type should be one of [%s], you provided %s." %
            (", ".join(CTC), model_type))
    return CTC[model_type]
