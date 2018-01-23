#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


def identity_initializer(scale=1.0):
    def _initializer(shape, dtype=tf.float32):
        if len(shape) == 1:
            return tf.constant(0., dtype=dtype, shape=shape)
        elif len(shape) == 2 and shape[0] == shape[1]:
            return tf.constant(scale * np.identity(shape[0], dtype))
        elif len(shape) == 4 and shape[2] == shape[3]:
            array = np.zeros(shape, dtype=float)
            cx, cy = shape[0] / 2, shape[1] / 2
            for i in range(shape[2]):
                array[cx, cy, i, i] = 1
            return tf.constant(scale * array, dtype=dtype)
        else:
            raise
    return _initializer


def orthogonal_initializer(scale=1.0):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        if partition_info is not None:
            ValueError("Do not know what to do with partition_info in BN_LSTMCell")
        flat_shape = (shape[0], int(np.prod(shape[1:])))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)  # this needs to be corrected to float32
        # return tf.constant(scale * q[:shape[0], :shape[1]], dtype=dtype)
        return tf.constant(scale * q, dtype=dtype)

    return _initializer
