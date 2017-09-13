#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Multi-task (bidirectional) LSTM-CTC model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.ctc.base_multitask import multitaskCTCBase
from models.encoders.multitask_blstm_encoder import Multitask_BLSTM_Encoder
from models.encoders.multitask_lstm_encoder import Multitask_LSTM_Encoder


class Multitask_LSTM_CTC(multitaskCTCBase):
    """Multi-task (bidirectional) LSTM-CTC model.
    Args:

        input_size (int): the dimensions of input vectors
        num_units (int): the number of units in each layer
        num_layers_main (int): the number of layers of the main task
        num_layers_sub (int): the number of layers of the sub task
        num_classes_main (int): the number of classes of target labels in the
            main task (except for a blank label)
        num_classes_second (int): the number of classes of target labels in the
            second task (except for a blank label)
        main_task_weight: A float value. The weight of loss of the main task.
            Set between 0 to 1
        bidirectional (bool): if True, create a bidirectional model
        lstm_impl (string, optional): BasicLSTMCell or LSTMCell or or LSTMBlockCell or
            LSTMBlockFusedCell.
            Choose the background implementation of tensorflow.
            Default is LSTMBlockCell (the fastest implementation).
        use_peephole (bool, optional): if True, use peephole
        splice (int, optional): frames to splice. Default is 1 frame.
        parameter_init (float, optional): Range of uniform distribution to
            initialize weight parameters (>= 0)
        clip_grad (float): the range of gradient clipping (> 0)
        clip_activation (float, optional): Range of activation clipping (> 0)
        num_proj (int, optional): the number of nodes in recurrent projection layer
        weight_decay (float, optional): a parameter for weight decay
        bottleneck_dim (int, optional): the dimensions of the bottleneck layer
    """

    def __init__(self,
                 input_size,
                 num_units,
                 num_layers_main,
                 num_layers_sub,
                 num_classes_main,
                 num_classes_sub,
                 main_task_weight,
                 bidirectional,
                 lstm_impl='LSTMBlockCell',
                 use_peephole=True,
                 splice=1,
                 parameter_init=0.1,
                 clip_grad=None,
                 clip_activation=None,
                 num_proj=None,
                 weight_decay=0.0,
                 bottleneck_dim=None):

        super(Multitask_LSTM_CTC, self).__init__(
            input_size, splice, num_classes_main, num_classes_sub,
            main_task_weight, clip_grad, weight_decay)

        if bidirectional:
            self.name = 'multitask_blstm_ctc'
            self.encoder = Multitask_BLSTM_Encoder(num_units,
                                                   num_layers_main,
                                                   num_layers_sub,
                                                   num_classes_main + 1,
                                                   num_classes_sub + 1,
                                                   lstm_impl,
                                                   use_peephole,
                                                   parameter_init,
                                                   clip_activation,
                                                   num_proj,
                                                   bottleneck_dim)
        else:
            self.name = 'multitask_lstm_ctc'
            self.encoder = Multitask_LSTM_Encoder(num_units,
                                                  num_layers_main,
                                                  num_layers_sub,
                                                  num_classes_main + 1,
                                                  num_classes_sub + 1,
                                                  lstm_impl,
                                                  use_peephole,
                                                  parameter_init,
                                                  clip_activation,
                                                  num_proj,
                                                  bottleneck_dim)
