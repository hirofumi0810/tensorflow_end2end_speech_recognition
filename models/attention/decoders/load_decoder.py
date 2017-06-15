#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Select & load decoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class load(object):
    """Select & load model.
    Args:
        model_type: string, lstm or gru
    Returns:
        model: An instance of the RNN
    """

    def __init__(self, model_type):
        if model_type not in ['lstm_decoder', 'gru_decoder']:
            raise ValueError(
                'model_type should be one of ["lstm_decoder", "gru_decoder"], you provided %s.' %
                (model_type))

        self.model_type = model_type

    def __call__(self, parameter_init, num_unit, clip_activation=None):
        """
        Args:
            parameter_init: A float value. Range of uniform distribution to
                initialize weight parameters
            num_unit: int, the number of units in each layer of the
                decoder
            clip_activation: A float value. Range of activation clipping (> 0)
        """
        if self.model_type == 'lstm_decoder':
            with tf.name_scope('lstm_decoder'):
                initializer = tf.random_uniform_initializer(
                    minval=-parameter_init,
                    maxval=parameter_init)
                lstm_decoder = tf.contrib.rnn.LSTMCell(
                    num_units=num_unit,
                    use_peepholes=True,
                    cell_clip=clip_activation,
                    initializer=initializer,
                    num_proj=None,
                    forget_bias=1.0,
                    state_is_tuple=True)

                return lstm_decoder

        elif self.model_type == 'gru_decoder':
            with tf.name_scope('gru_decoder'):
                initializer = tf.random_uniform_initializer(
                    minval=-parameter_init,
                    maxval=parameter_init)

                with tf.variable_scope('gru', initializer=initializer):
                    gru_decoder = tf.contrib.rnn.GRUCell(num_unit)

                    return gru_decoder
