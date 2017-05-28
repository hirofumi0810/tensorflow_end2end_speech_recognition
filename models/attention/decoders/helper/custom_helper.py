#! /usr/bin/env python
# -*- coding: utf-8 -*-

from helper_base import Helper
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops


class CustomHelper(Helper):
    """Base abstract class that allows the user to customize sampling."""

    def __init__(self, initialize_fn, sample_fn, next_inputs_fn):
        """Initializer.
        Args:
          initialize_fn: callable that returns `(finished, next_inputs)`
            for the first iteration.
          sample_fn: callable that takes `(time, outputs, state)`
            and emits tensor `sample_ids`.
          next_inputs_fn: callable that takes `(time, outputs, state, sample_ids)`
            and emits `(finished, next_inputs, next_state)`.
        """
        self._initialize_fn = initialize_fn
        self._sample_fn = sample_fn
        self._next_inputs_fn = next_inputs_fn
        self._batch_size = None

    @property
    def batch_size(self):
        if self._batch_size is None:
            raise ValueError(
                "batch_size accessed before initialize was called")
        return self._batch_size

    def initialize(self, name=None):
        with ops.name_scope(name, "%sInitialize" % type(self).__name__):
            (finished, next_inputs) = self._initialize_fn()
            if self._batch_size is None:
                self._batch_size = array_ops.size(finished)
        return (finished, next_inputs)

    def sample(self, time, outputs, state, name=None):
        with ops.name_scope(
                name, "%sSample" % type(self).__name__, (time, outputs, state)):
            return self._sample_fn(time=time, outputs=outputs, state=state)

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        with ops.name_scope(
                name, "%sNextInputs" % type(self).__name__, (time, outputs, state)):
            return self._next_inputs_fn(
                time=time, outputs=outputs, state=state, sample_ids=sample_ids)
