#! /usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf


def batch_norm(inputs, name_scope, is_training, epsilon=1e-3, decay=0.99):
    with tf.variable_scope(name_scope):
        size = inputs.get_shape().as_list()[1]

        gamma = tf.get_variable(
            'gamma', [size], initializer=tf.constant_initializer(0.1))
        # beta = tf.get_variable('beta', [size], initializer=tf.constant_initializer(0))
        beta = tf.get_variable('beta', [size])

        pop_mean = tf.get_variable('pop_mean', [size],
                                   initializer=tf.zeros_initializer(), trainable=False)
        pop_var = tf.get_variable('pop_var', [size],
                                  initializer=tf.ones_initializer(), trainable=False)
        batch_mean, batch_var = tf.nn.moments(inputs, [0])

        train_mean_op = tf.assign(
            pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var_op = tf.assign(
            pop_var, pop_var * decay + batch_var * (1 - decay))

        def batch_statistics():
            with tf.control_dependencies([train_mean_op, train_var_op]):
                return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, gamma, epsilon)

        def pop_statistics():
            return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, gamma, epsilon)

        # control flow
        return tf.cond(is_training, batch_statistics, pop_statistics)
