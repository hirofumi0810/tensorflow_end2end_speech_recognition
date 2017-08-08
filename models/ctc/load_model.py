#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Select & load CTC model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.ctc.layers.lstm_ctc import LSTM_CTC
from models.ctc.layers.blstm_ctc import BLSTM_CTC
from models.ctc.layers.gru_ctc import GRU_CTC
from models.ctc.layers.bgru_ctc import BGRU_CTC
from models.ctc.layers.cnn_ctc import CNN_CTC
from models.ctc.layers.vgg_blstm_ctc import VGG_BLSTM_CTC
from models.ctc.layers.multitask_blstm_ctc import Multitask_BLSTM_CTC


CTC = {
    "lstm_ctc": LSTM_CTC,
    "blstm_ctc": BLSTM_CTC,
    "gru_ctc": GRU_CTC,
    "bgru_ctc": BGRU_CTC,
    "cnn_ctc": CNN_CTC,
    "vgg_blstm_ctc": VGG_BLSTM_CTC,
    "multitask_blstm_ctc": Multitask_BLSTM_CTC
}


def load(model_type):
    """Select & load model.
    Args:
        model_type: string, name of the ctc model in the key of CTC
    Returns:
        model: class object
    """
    if model_type not in CTC:
        raise ValueError(
            "model_type should be one of [%s], you provided %s." %
            (", ".join(CTC), model_type))

    return CTC[model_type]
