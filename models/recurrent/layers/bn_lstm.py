#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Long-Short Term Memory with Batch Normalization."""

import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell, LSTMStateTuple
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.platform import tf_logging as logging
from .batch_normalization import batch_norm


class BatchNormLSTMCell(RNNCell):
    """Batch Normalized Long short-term memory unit (LSTM) recurrent network cell.
    The default non-peephole implementation is based on:
      http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
    S. Hochreiter and J. Schmidhuber.
    "Long Short-Term Memory". Neural Computation, 9(8):1735-1780, 1997.
    The peephole implementation is based on:
      https://research.google.com/pubs/archive/43905.pdf
    Hasim Sak, Andrew Senior, and Francoise Beaufays.
    "Long short-term memory recurrent neural network architectures for
     large scale acoustic modeling." INTERSPEECH, 2014.
    The class uses optional peep-hole connections, optional cell clipping, and
    an optional projection layer.
    """

    def __init__(self, num_units, is_training, input_size=None,
                 use_peepholes=False, cell_clip=None,
                 initializer=None, num_proj=None, proj_clip=None,
                 num_unit_shards=None, num_proj_shards=None,
                 forget_bias=1.0, state_is_tuple=True,
                 reuse=None):
        """Initialize the parameters for an LSTM cell.
        Args:
          num_units: int, The number of units in the LSTM cell
          is_training: bool, set True when training.
          input_size: Deprecated and unused.
          use_peepholes: bool, set True to enable diagonal/peephole connections.
          cell_clip: (optional) A float value, if provided the cell state is clipped
            by this value prior to the cell output activation.
          initializer: (optional) The initializer to use for the weight and
            projection matrices.
          num_proj: (optional) int, The output dimensionality for the projection
            matrices.  If None, no projection is performed.
          proj_clip: (optional) A float value.  If `num_proj > 0` and `proj_clip` is
            provided, then the projected values are clipped elementwise to within
            `[-proj_clip, proj_clip]`.
          num_unit_shards: Deprecated, will be removed by Jan. 2017.
            Use a variable_scope partitioner instead.
          num_proj_shards: Deprecated, will be removed by Jan. 2017.
            Use a variable_scope partitioner instead.
          forget_bias: Biases of the forget gate are initialized by default to 1
            in order to reduce the scale of forgetting at the beginning of
            the training.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  This latter behavior will soon be deprecated.
          reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.  If not `True`, and the existing scope already has
            the given variables, an error is raised.
        """
        if not state_is_tuple:
            logging.warn("%s: Using a concatenated state is slower and will soon be "
                         "deprecated.  Use state_is_tuple=True.", self)
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        if num_unit_shards is not None or num_proj_shards is not None:
            logging.warn("%s: The num_unit_shards and proj_unit_shards parameters are "
                         "deprecated and will be removed in Jan 2017.  "
                         "Use a variable scope with a partitioner instead.", self)

        self._num_units = num_units
        self._use_peepholes = use_peepholes
        self._cell_clip = cell_clip
        self._initializer = initializer
        self._num_proj = num_proj
        self._proj_clip = proj_clip
        self._num_unit_shards = num_unit_shards
        self._num_proj_shards = num_proj_shards
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._reuse = reuse
        self._is_training = is_training

        if num_proj:
            self._state_size = (LSTMStateTuple(num_units, num_proj)
                                if state_is_tuple else num_units + num_proj)
            self._output_size = num_proj
        else:
            self._state_size = (LSTMStateTuple(num_units, num_units)
                                if state_is_tuple else 2 * num_units)
            self._output_size = num_units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state, scope=None):
        """Run one step of LSTM.
        Args:
          inputs: input Tensor, 2D, batch x num_units.
          state: if `state_is_tuple` is False, this must be a state Tensor,
            `2-D, batch x state_size`.  If `state_is_tuple` is True, this must be a
            tuple of state Tensors, both `2-D`, with column sizes `c_state` and
            `m_state`.
          scope: VariableScope for the created subgraph; defaults to "lstm_cell".
        Returns:
          A tuple containing:
          - A `2-D, [batch x output_dim]`, Tensor representing the output of the
            LSTM after reading `inputs` when previous state was `state`.
            Here output_dim is:
               num_proj if num_proj was set,
               num_units otherwise.
          - Tensor(s) representing the new state of LSTM after reading `inputs` when
            the previous state was `state`.  Same type and shape(s) as `state`.
        Raises:
          ValueError: If input size cannot be inferred from inputs via
            static shape inference.
        """
        num_proj = self._num_units if self._num_proj is None else self._num_proj

        if self._state_is_tuple:
            (c_prev, h_prev) = state
        else:
            c_prev = tf.slice(state, begin=[0, 0], size=[-1, self._num_units])
            h_prev = tf.slice(
                state, begin=[0, self._num_units], size=[-1, num_proj])

        dtype = inputs.dtype
        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError(
                "Could not infer input size from inputs.get_shape()[-1]")

        with tf.variable_scope(scope or "lstm_cell", initializer=self._initializer,
                               reuse=self._reuse) as unit_scope:
            if self._num_unit_shards is not None:
                unit_scope.set_partitioner(
                    partitioned_variables.fixed_size_partitioner(
                        self._num_unit_shards))

            W_xh = tf.get_variable('W_xh', shape=[input_size, 4 * self._num_units],
                                   initializer=self._initializer)
            W_hh = tf.get_variable('W_hh', shape=[num_proj, 4 * self._num_units],
                                   initializer=self._initializer)
            bias = tf.get_variable('b', [4 * self._num_units])

            xh = tf.matmul(inputs, W_xh)
            hh = tf.matmul(h_prev, W_hh)

            bn_xh = batch_norm(xh, 'xh', self._is_training)
            bn_hh = batch_norm(hh, 'hh', self._is_training)

            # i = input_gate, g = new_input, f = forget_gate, o = output_gate
            # lstm_matrix = tf.contrib.rnn._linear([inputs, h_prev], 4 * self._num_units, bias=True)
            lstm_matrix = tf.nn.bias_add(tf.add(bn_xh, bn_hh), bias)
            i, g, f, o = tf.split(
                value=lstm_matrix, num_or_size_splits=4, axis=1)

            # Diagonal connections
            if self._use_peepholes:
                # tf.variable_scopeとtf.get_variableはセットで使う
                with tf.variable_scope(unit_scope) as projection_scope:
                    if self._num_unit_shards is not None:
                        projection_scope.set_partitioner(None)
                    p_f_diag = tf.get_variable(
                        "p_f_diag", shape=[self._num_units], dtype=dtype)
                    p_i_diag = tf.get_variable(
                        "p_i_diag", shape=[self._num_units], dtype=dtype)
                    p_o_diag = tf.get_variable(
                        "p_o_diag", shape=[self._num_units], dtype=dtype)

            if self._use_peepholes:
                c = (tf.sigmoid(f + self._forget_bias + p_f_diag * c_prev) * c_prev +
                     tf.sigmoid(i + p_i_diag * c_prev) * tf.tanh(g))
            else:
                c = (tf.sigmoid(f + self._forget_bias) * c_prev +
                     tf.sigmoid(i) * tf.tanh(g))

            if self._cell_clip is not None:
                c = tf.clip_by_value(c, -self._cell_clip, self._cell_clip)

            bn_c = batch_norm(c, 'bn_c', self._is_training)

            if self._use_peepholes:
                # peephole側には適用しない
                h = tf.sigmoid(o + p_o_diag * c) * tf.tanh(bn_c)
            else:
                h = tf.sigmoid(o) * tf.tanh(bn_c)

            if self._num_proj is not None:
                with tf.variable_scope("projection") as proj_scope:
                    if self._num_proj_shards is not None:
                        proj_scope.set_partitioner(
                            partitioned_variables.fixed_size_partitioner(
                                self._num_proj_shards))
                    h = tf.contrib.rnn._linear(h, self._num_proj, bias=False)

                if self._proj_clip is not None:
                    h = tf.clip_by_value(h, -self._proj_clip, self._proj_clip)

        new_state = (LSTMStateTuple(c, h) if self._state_is_tuple else
                     tf.concat(values=[c, h], axis=1))
        return h, new_state
