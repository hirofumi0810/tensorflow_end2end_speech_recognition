#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Quasi Recurrent Neural Network."""

import tensorflow as tf


class QRNN(object):
    """Quasi-Recurrent Neural Networks.
        See details in https://arxiv.org/abs/1611.01576.
    """

    def __init__(self, in_size, size, conv_size=2):
        """
        Args:
            in_size:
            size:
            conv_size:
        """
        self.kernel = None
        self.batch_size = -1
        self.conv_size = conv_size
        self.c = None
        self.h = None
        self._x = None
        if conv_size == 1:
            self.kernel = QRNNLinear(in_size, size)
        elif conv_size == 2:
            self.kernel = QRNNWithPrevious(in_size, size)
        else:
            self.kernel = QRNNConvolution(in_size, size, conv_size)

    def _step(self, f, z, o):
        """
        Args:
            f:
            z:
            o:
        Returns:
            h:
        """
        with tf.variable_scope("fo-Pool"):
            # f,z,o is batch_size x size
            f = tf.sigmoid(f)
            z = tf.tanh(z)
            o = tf.sigmoid(o)
            self.c = tf.mul(f, self.c) + tf.mul(1 - f, z)
            self.h = tf.mul(o, self.c)  # h is size vector

        return self.h

    def forward(self, x):
        """
        Args:
            x:
        Returns:
            h:
        """
        def length(mx): return int(mx.get_shape()[0])

        with tf.variable_scope("QRNN/Forward"):
            if self.c is None:
                # init context cell
                self.c = tf.zeros(
                    [length(x), self.kernel.size], dtype=tf.float32)

            if self.conv_size <= 2:
                # x is batch_size x sentence_length x word_length
                # -> now, transpose it to sentence_length x batch_size x word_length
                _x = tf.transpose(x, [1, 0, 2])

                for i in range(length(_x)):
                    t = _x[i]  # t is batch_size x word_length matrix
                    f, z, o = self.kernel.forward(t)
                    self._step(f, z, o)
            else:
                c_f, c_z, c_o = self.kernel.conv(x)
                for i in range(length(c_f)):
                    f, z, o = c_f[i], c_z[i], c_o[i]
                    self._step(f, z, o)

        return self.h


class QRNNLinear():
    """
    """

    def __init__(self, in_size, size):
        """
        Args:
            in_size:
            size:
        """
        self.in_size = in_size
        self.size = size
        self._weight_size = self.size * 3  # z, f, o
        with tf.variable_scope("QRNN/Variable/Linear"):
            initializer = tf.random_normal_initializer()
            self.W = tf.get_variable(
                "W", [self.in_size, self._weight_size],
                initializer=initializer)
            self.b = tf.get_variable(
                "b", [self._weight_size], initializer=initializer)

    def forward(self, t):
        # x is batch_size x word_length matrix
        _weighted = tf.matmul(t, self.W)
        _weighted = tf.add(_weighted, self.b)

        # now, _weighted is batch_size x weight_size
        # split to f, z, o. each matrix is batch_size x size
        f, z, o = tf.split(1, 3, _weighted)
        return f, z, o


class QRNNWithPrevious():

    def __init__(self, in_size, size):
        """
        Args:
            in_size:
            size:
        """
        self.in_size = in_size
        self.size = size
        self._weight_size = self.size * 3  # z, f, o
        self._previous = None
        with tf.variable_scope("QRNN/Variable/WithPrevious"):
            initializer = tf.random_normal_initializer()
            self.W = tf.get_variable(
                "W", [self.in_size, self._weight_size],
                initializer=initializer)
            self.V = tf.get_variable(
                "V", [self.in_size, self._weight_size],
                initializer=initializer)
            self.b = tf.get_variable(
                "b", [self._weight_size], initializer=initializer)

    def forward(self, t):
        if self._previous is None:
            self._previous = tf.get_variable(
                "previous", [t.get_shape()[0], self.in_size],
                initializer=tf.random_normal_initializer())

        _current = tf.matmul(t, self.W)
        _previous = tf.matmul(self._previous, self.V)
        _previous = tf.add(_previous, self.b)
        _weighted = tf.add(_current, _previous)

        # split to f, z, o. each matrix is batch_size x size
        f, z, o = tf.split(1, 3, _weighted)
        self._previous = t
        return f, z, o


class QRNNConvolution():

    def __init__(self, in_size, size, conv_size):
        """
        Args:
            in_size:
            size:
            conv_size:
        """
        self.in_size = in_size
        self.size = size
        self.conv_size = conv_size
        self._weight_size = self.size * 3  # z, f, o

        with tf.variable_scope("QRNN/Variable/Convolution"):
            initializer = tf.random_normal_initializer()
            self.conv_filter = tf.get_variable(
                "conv_filter", [conv_size, in_size, self._weight_size],
                initializer=initializer)

    def conv(self, x):
        # !! x is batch_size x sentence_length x word_length(=channel) !!
        _weighted = tf.nn.conv1d(
            x, self.conv_filter, stride=1, padding="SAME", data_format="NHWC")

        # _weighted is batch_size x conved_size x output_channel
        # conved_size x  batch_size x output_channel
        _w = tf.transpose(_weighted, [1, 0, 2])
        # make 3(f, z, o) conved_size x  batch_size x size
        _ws = tf.split(2, 3, _w)
        return _ws
