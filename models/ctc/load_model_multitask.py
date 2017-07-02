#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Select & load Multitask CTC model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.ctc.multitask_blstm_ctc import Multitask_BLSTM_CTC

CTC = {
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
