#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Select & load CTC model."""


from .lstm_ctc import LSTM_CTC
from .blstm_ctc import BLSTM_CTC
from .gru_ctc import GRU_CTC
from .bgru_ctc import BGRU_CTC
from .cnn_ctc import CNN_CTC


def load(model_type):
    """Select & load model.
    Args:
        model_type: lstm or blstm or gru or bgru or cnn
    Returns:
        model: class object
    """
    if model_type == 'lstm_ctc':
        return LSTM_CTC
    elif model_type == 'blstm_ctc':
        return BLSTM_CTC
    elif model_type == 'gru_ctc':
        return GRU_CTC
    elif model_type == 'bgru_ctc':
        return BGRU_CTC
    elif model_type == 'cnn_ctc':
        return CNN_CTC
    else:
        raise ValueError(
            'Model is "lstm_ctc" or "blstm_ctc" or "gru_ctc" or "bgru_ctc" or "cnn_ctc".')
