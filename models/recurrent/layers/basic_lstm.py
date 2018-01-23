#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Bisic Long-Short Term Memory (no peep-hole connections).
   This code is taken directly from TensorFlow code (some functions are modified).
"""

import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell, LSTMStateTuple
from tensorflow.python.platform import tf_logging as logging


class BasicLSTMCell(RNNCell):
    """Basic LSTM recurrent network cell.
    The implementation is based on: http://arxiv.org/abs/1409.2329.
    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.
    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.
    For advanced models, please use the full LSTMCell that follows.
    """

    def __init__(self, num_units, forget_bias=1.0, input_size=None,
                 state_is_tuple=True, reuse=None):
        """Initialize the basic LSTM cell.
        Args:
          num_units: int, The number of units in the LSTM cell.
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

    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(self, scope or "basic_lstm_cell", reuse=self._reuse):
            # Parameters of gates are concatenated into one multiply for
            # efficiency.
            if self._state_is_tuple:
                c_prev, h_prev = state
            else:
                c_prev, h_prev = tf.split(
                    value=state, num_or_size_splits=2, axis=1)
            concat = tf.contrib.rnn._linear(
                [inputs, h_prev], 4 * self._num_units, True)

            # i = input_gate, g = new_input, f = forget_gate, o = output_gate
            i, g, f, o = tf.split(value=concat, num_or_size_splits=4, axis=1)

            c = (c_prev * tf.sigmoid(f + self._forget_bias) +
                 tf.sigmoid(i) * tf.tanh(g))
            h = tf.tanh(c) * tf.sigmoid(o)

            if self._state_is_tuple:
                new_state = LSTMStateTuple(c, h)
            else:
                new_state = tf.concat([c, h], 1)
            return h, new_state
