#! /usr/bin/env python
# -*- coding: utf-8 -*-

import abc


@six.add_metaclass(abc.ABCMeta)
class Helper(object):
    """Interface for implementing sampling in seq2seq decoders.
    Helper instances are used by `BasicDecoder`.
    """

    @abc.abstractproperty
    def batch_size(self):
        """Batch size of tensor returned by `sample`.
        Returns a scalar int32 tensor.
        """
        raise NotImplementedError("batch_size has not been implemented")

    @abc.abstractmethod
    def initialize(self, name=None):
        """Returns `(initial_finished, initial_inputs)`."""
        pass

    @abc.abstractmethod
    def sample(self, time, outputs, state, name=None):
        """Returns `sample_ids`."""
        pass

    @abc.abstractmethod
    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """Returns `(finished, next_inputs, next_state)`."""
        pass
