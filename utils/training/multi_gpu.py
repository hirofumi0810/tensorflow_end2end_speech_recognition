#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilities for mulit-GPU implementation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def average_gradients(total_grads_and_vars):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
        total_grads_and_vars: List of lists of (gradient, variable) tuples.
            The outer list is over individual gradients. The inner list is over
            the gradient calculation for each tower.
    Returns:
        average_grads_and_vars: List of pairs of (gradient, variable) where
            the gradient has been averaged across all towers.
    """
    average_grads_and_vars = []
    for tower_grads_and_vars in zip(*total_grads_and_vars):
        # Note that each tower_grads_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        tower_grads = []
        for grad, _ in tower_grads_and_vars:  # var is ignored
            if grad is not None:
                # Add 0 dimension to the gradients to represent the tower
                expanded_grad = tf.expand_dims(grad, axis=0)

                # Append on a 'tower' dimension which we will average over
                # below
                tower_grads.append(expanded_grad)

        # Average over the 'tower' dimension.
        tower_grads = tf.concat(axis=0, values=tower_grads)
        mean_tower_grad = tf.reduce_mean(tower_grads, axis=0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        var = tower_grads_and_vars[0][1]  # var0_gpu0
        grad_and_var = (mean_tower_grad, var)
        average_grads_and_vars.append(grad_and_var)
    return average_grads_and_vars
