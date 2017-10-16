#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class for all models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

OPTIMIZER_CLS_NAMES = {
    "adagrad": tf.train.AdagradOptimizer,
    "adadelta": tf.train.AdadeltaOptimizer,
    "adam": tf.train.AdamOptimizer,
    "rmsprop": tf.train.RMSPropOptimizer,
    "sgd": tf.train.GradientDescentOptimizer,
    "momentum": tf.train.MomentumOptimizer,
    "nestrov": tf.train.MomentumOptimizer
}


class ModelBase(object):

    def __init__(self, *args, **kwargs):
        pass

    def _build(self, *args, **kwargs):
        """Construct model graph."""
        raise NotADirectoryError

    def create_placeholders(self):
        """Create placeholders and append them to list."""
        raise NotImplementedError

    def compute_loss(self, *args, **kwargs):
        """Operation for computing loss."""
        raise NotImplementedError

    def _add_noise_to_inputs(self, inputs, stddev=0.075):
        """Add gaussian noise to the inputs.
        Args:
            inputs: the noise free input-features.
            stddev (float, optional): The standart deviation of the noise.
                Default is 0.075.
        Returns:
            inputs: Input features plus noise.
        """
        # if stddev != 0:
        #     with tf.variable_scope("input_noise"):
        #         # Add input noise with a standart deviation of stddev.
        #         inputs = tf.random_normal(
        #             tf.shape(inputs), 0.0, stddev) + inputs
        # return inputs
        raise NotImplementedError

    def _add_noise_to_gradients(grads_and_vars, gradient_noise_scale,
                                stddev=0.075):
        """Adds scaled noise from a 0-mean normal distribution to gradients.
        Args:
            grads_and_vars:
            gradient_noise_scale:
            stddev (float):
        Returns:
        """
        raise NotImplementedError

    def _set_optimizer(self, optimizer, learning_rate):
        """Set optimizer.
        Args:
            optimizer (string): the name of the optimizer in
                OPTIMIZER_CLS_NAMES
            learning_rate (float): A learning rate
        Returns:
            optimizer:
        """
        optimizer = optimizer.lower()
        if optimizer not in OPTIMIZER_CLS_NAMES:
            raise ValueError(
                "Optimizer name should be one of [%s], you provided %s." %
                (", ".join(OPTIMIZER_CLS_NAMES), optimizer))

        # Select optimizer
        if optimizer == 'momentum':
            return OPTIMIZER_CLS_NAMES[optimizer](
                learning_rate=learning_rate,
                momentum=0.9)
        elif optimizer == 'nestrov':
            return OPTIMIZER_CLS_NAMES[optimizer](
                learning_rate=learning_rate,
                momentum=0.9,
                use_nesterov=True)
        else:
            return OPTIMIZER_CLS_NAMES[optimizer](
                learning_rate=learning_rate)

    def train(self, loss, optimizer, learning_rate):
        """Operation for training. Only the sigle GPU training is supported.
        Args:
            loss: An operation for computing loss
            optimizer (string): name of the optimizer in OPTIMIZER_CLS_NAMES
            learning_rate (placeholder): A learning rate
        Returns:
            train_op: operation for training
        """
        # Create a variable to track the global step
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # Set optimizer
        self.optimizer = self._set_optimizer(optimizer, learning_rate)

        if self.clip_grad_norm is not None:
            # Compute gradients
            grads_and_vars = self.optimizer.compute_gradients(loss)

            # Clip gradients
            clipped_grads_and_vars = self._clip_gradients(grads_and_vars)

            # Create operation for gradient update
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = self.optimizer.apply_gradients(
                    clipped_grads_and_vars,
                    global_step=global_step)

        else:
            # Use the optimizer to apply the gradients that minimize the loss
            # and also increment the global step counter as a single training
            # step
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = self.optimizer.minimize(
                    loss, global_step=global_step)

        return train_op

    def _clip_gradients(self, grads_and_vars):
        """Clip gradients.
        Args:
            grads_and_vars (list): list of tuples of `(grads, vars)`
        Returns:
            clipped_grads_and_vars (list): list of tuple of
                `(clipped grads, vars)`
        """
        # TODO: Optionally add gradient noise

        clipped_grads_and_vars = []

        # Clip gradient norm
        for grad, var in grads_and_vars:
            if grad is not None:
                clipped_grads_and_vars.append(
                    (tf.clip_by_norm(grad, clip_norm=self.clip_grad_norm),
                     var))

        # Clip gradient
        # for grad, var in grads_and_vars:
        #     if grad is not None:
        #         clipped_grads_and_vars.append(
        #             (tf.clip_by_value(grad,
        #                               clip_value_min=-self.clip_grad_norm,
        #                               clip_value_max=self.clip_grad_norm),
        #              var))

        # TODO: Add histograms for variables, gradients (norms)
        # self._tensorboard(trainable_vars)

        return clipped_grads_and_vars

    def _tensorboard(self, trainable_vars):
        """Compute statistics for TensorBoard plot.
        Args:
            trainable_vars:
        """
        ##############################
        # train
        ##############################
        with tf.name_scope("train"):
            for var in trainable_vars:
                self.summaries_train.append(
                    tf.summary.histogram(var.name, var))

        # Mean
        with tf.name_scope("mean_train"):
            for var in trainable_vars:
                self.summaries_train.append(
                    tf.summary.scalar(var.name, tf.reduce_mean(var)))

        # Standard deviation
        with tf.name_scope("stddev_train"):
            for var in trainable_vars:
                self.summaries_train.append(
                    tf.summary.scalar(var.name, tf.sqrt(
                        tf.reduce_mean(tf.square(var - tf.reduce_mean(var))))))

        # Max
        with tf.name_scope("max_train"):
            for var in trainable_vars:
                self.summaries_train.append(
                    tf.summary.scalar(var.name, tf.reduce_max(var)))

        # Min
        with tf.name_scope("min_train"):
            for var in trainable_vars:
                self.summaries_train.append(
                    tf.summary.scalar(var.name, tf.reduce_min(var)))

        ##############################
        # dev
        ##############################
        with tf.name_scope("dev"):
            for var in trainable_vars:
                self.summaries_dev.append(
                    tf.summary.histogram(var.name, var))

        # Mean
        with tf.name_scope("mean_dev"):
            for var in trainable_vars:
                self.summaries_dev.append(
                    tf.summary.scalar(var.name, tf.reduce_mean(var)))

        # Standard deviation
        with tf.name_scope("stddev_dev"):
            for var in trainable_vars:
                self.summaries_dev.append(
                    tf.summary.scalar(var.name, tf.sqrt(
                        tf.reduce_mean(tf.square(var - tf.reduce_mean(var))))))

        # Max
        with tf.name_scope("max_dev"):
            for var in trainable_vars:
                self.summaries_dev.append(
                    tf.summary.scalar(var.name, tf.reduce_max(var)))

        # Min
        with tf.name_scope("min_dev"):
            for var in trainable_vars:
                self.summaries_dev.append(
                    tf.summary.scalar(var.name, tf.reduce_min(var)))
