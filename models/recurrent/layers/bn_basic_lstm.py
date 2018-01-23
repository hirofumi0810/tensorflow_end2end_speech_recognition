#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Basic Long-Short Term Memory with Batch Normalization."""

import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell, LSTMStateTuple
from tensorflow.python.platform import tf_logging as logging
from .batch_normalization import batch_norm
from .initializer import orthogonal_initializer


class BatchNormBasicLSTMCell(RNNCell):
    """Batch Normalized Basic LSTM recurrent network cell.
    The implementation is based on: http://arxiv.org/abs/1409.2329.
    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.
    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.
    For advanced models, please use the full LSTMCell that follows.
    """

    def __init__(self, num_units, is_training, forget_bias=1.0, input_size=None,
                 state_is_tuple=True, reuse=None):
        """Initialize the basic LSTM cell.
        Args:
          num_units: int, The number of units in the LSTM cell.
          is_training: bool, set True when training.
          forget_bias: float, The bias added to forget gates (see above).
          input_size: Deprecated and unused.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  The latter behavior will soon be deprecated.
          reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.  If not `True`, and the existing scope already has
            the given variables, an error is raised.
        """
        if not state_is_tuple:
            logging.warn("%s: Using a concatenated state is slower and will soon be "
                         "deprecated.  Use state_is_tuple=True.", self)
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._reuse = reuse
        self._is_training = is_training

    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM) with Recurrent Batch Normalization."""
        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError(
                "Could not infer input size from inputs.get_shape()[-1]")

        with tf.variable_scope(scope or "batch_norm_lstm_cell", reuse=self._reuse):
            # Parameters of gates are concatenated into one multiply for
            # efficiency.
            if self._state_is_tuple:
                c_prev, h_prev = state
            else:
                c_prev, h_prev = tf.split(
                    value=state, num_or_size_splits=2, axis=1)

            W_xh = tf.get_variable('W_xh', shape=[input_size, 4 * self._num_units],
                                   initializer=orthogonal_initializer())
            W_hh = tf.get_variable('W_hh', shape=[self._num_units, 4 * self._num_units],
                                   initializer=orthogonal_initializer())
            bias = tf.get_variable('b', [4 * self._num_units])

            xh = tf.matmul(inputs, W_xh)
            hh = tf.matmul(h_prev, W_hh)

            bn_xh = batch_norm(xh, 'xh', self._is_training)
            bn_hh = batch_norm(hh, 'hh', self._is_training)

            # i = input_gate, g = new_input, f = forget_gate, o = output_gate
            # lstm_matrix = tf.contrib.rnn._linear([inputs, h_prev], 4 * self._num_units, True)
            lstm_matrix = tf.nn.bias_add(tf.add(bn_xh, bn_hh), bias)
            i, g, f, o = tf.split(
                value=lstm_matrix, num_or_size_splits=4, axis=1)

            c = (c_prev * tf.sigmoid(f + self._forget_bias) +
                 tf.sigmoid(i) * tf.tanh(g))

            bn_c = batch_norm(c, 'bn_c', self._is_training)

            h = tf.tanh(bn_c) * tf.sigmoid(o)

            if self._state_is_tuple:
                new_state = LSTMStateTuple(c, h)
            else:
                new_state = tf.concat(values=[c, h], axis=1)
            return h, new_state
