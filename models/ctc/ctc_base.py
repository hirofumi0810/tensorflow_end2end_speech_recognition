#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""CTC network."""

import tensorflow as tf


OPTIMIZER_CLS_NAMES = {
    "adagrad": tf.train.AdagradOptimizer,
    "adadelta": tf.train.AdadeltaOptimizer,
    "adam": tf.train.AdamOptimizer,
    "momentum": tf.train.MomentumOptimizer,
    "rmsprop": tf.train.RMSPropOptimizer,
    "sgd": tf.train.GradientDescentOptimizer,
}


class ctcBase(object):
    """Connectionist Temporal Classification (CTC) network.
    Args:
        batch_size: int, batch size of mini batch
        input_size: int, the dimensions of input vectors
        num_cell: int, the number of memory cells in each layer
        num_layers: int, the number of layers
        output_size: int, the number of nodes in softmax layer (except for blank class)
        parameter_init: A float value. Range of uniform distribution to initialize weight parameters
        clip_gradients: A float value. Range of gradient clipping (non-negative)
        clip_activation: A float value. Range of activation clipping (non-negative)
        dropout_ratio_input: A float value. Dropout ratio in input-hidden layers
        dropout_ratio_hidden: A float value. Dropout ratio in hidden-hidden layers
        weight_decay: A float value. Regularization parameter for weight decay
    """

    def __init__(self,
                 batch_size,
                 input_size,
                 num_cell,
                 num_layers,
                 output_size,
                 parameter_init,
                 clip_gradients,
                 clip_activation,
                 dropout_ratio_input,
                 dropout_ratio_hidden,
                 weight_decay):

        # Network size
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_cell = num_cell
        self.num_layers = num_layers
        self.num_classes = output_size + 1  # plus blank label

        # Regularization
        self.parameter_init = parameter_init
        self.clip_gradients = clip_gradients
        self.clip_activation = clip_activation
        if dropout_ratio_input == 1.0 and dropout_ratio_hidden == 1.0:
            self.dropout = False
        else:
            self.dropout = True
        self.dropout_ratio_input = dropout_ratio_input
        self.dropout_ratio_hidden = dropout_ratio_hidden
        self.weight_decay = weight_decay

        # Summaries for TensorBoard
        self.summaries_train = []
        self.summaries_dev = []

    def _generate_pl(self):
        """Generate placeholders."""
        # [batch_size, max_timesteps, input_size_splice]
        self.inputs_pl = tf.placeholder(tf.float32,
                                        shape=[None, None, self.input_size],
                                        name='input')
        # self.labels_pl = tf.sparse_placeholder(tf.int32, name='label')
        self.label_indices_pl = tf.placeholder(tf.int64, name='indices')
        self.label_values_pl = tf.placeholder(tf.int32, name='values')
        self.label_shape_pl = tf.placeholder(tf.int64, name='shape')
        self.labels_pl = tf.SparseTensor(self.label_indices_pl,
                                         self.label_values_pl,
                                         self.label_shape_pl)
        # [batch_size]
        self.seq_len_pl = tf.placeholder(tf.int64,
                                         shape=[None],
                                         name='seq_len')
        # NOTE: change to tf.int64 if you use the bidirectional model

        # For dropout
        self.keep_prob_input_pl = tf.placeholder(tf.float32,
                                                 name='keep_prob_input')
        self.keep_prob_hidden_pl = tf.placeholder(tf.float32,
                                                  name='keep_prob_hidden')

        # Learning rate
        self.lr_pl = tf.placeholder(tf.float32, name='learning_rate')

    def loss(self):
        """Operation for computing ctc loss.
        Returns:
            loss: operation for computing ctc loss
        """
        # Weight decay
        # weight_sum = 0
        # for var in tf.trainable_variables():
        #     if 'bias' not in var.name.lower():
        #         weight_sum += tf.nn.l2_loss(var)
        # tf.add_to_collection('losses', weight_sum * self.weight_decay)

        with tf.name_scope("ctc_loss"):
            ctc_loss = tf.nn.ctc_loss(
                self.labels_pl, self.logits, tf.cast(self.seq_len_pl, tf.int32))
            self.loss = tf.reduce_mean(ctc_loss, name='ctc_loss_mean')
            # tf.add_to_collection('losses', ctc_loss_mean)

            # print(ctc_loss_mean)
            # Total loss
            # self.loss = ctc_loss_mean + weight_decay * self.weight_decay
            # self.loss = ctc_loss_mean
            print(self.loss)

            # Add a scalar summary for the snapshot of loss
            self.summaries_train.append(
                tf.summary.scalar('loss_train', self.loss))
            self.summaries_dev.append(
                tf.summary.scalar('loss_dev', self.loss))

            return self.loss

    def train(self, optimizer, learning_rate_init=None, clip_gradients_by_norm=None, is_scheduled=False):
        """Operation for training.
        Args:
            optimizer: string, name of the optimizer in OPTIMIZER_CLS_NAMES
            learning_rate_init: initial learning rate
            clip_gradients_by_norm: if True, clip gradients by norm of the value of self.clip_gradients
            is_scheduled: if True, schedule learning rate at each epoch
        Returns:
            train_op: operation for training
        """
        optimizer = optimizer.lower()
        if optimizer not in OPTIMIZER_CLS_NAMES:
            raise ValueError(
                "Optimizer name should be one of [%s], you provided %s." %
                (", ".join(OPTIMIZER_CLS_NAMES), optimizer))
        if learning_rate_init < 0.0:
            raise ValueError("Invalid learning_rate %s.", learning_rate_init)

        # Select parameter update method
        if is_scheduled:
            learning_rate = self.lr_pl
        else:
            learning_rate = learning_rate_init

        if optimizer == 'momentum':
            self.optimizer = OPTIMIZER_CLS_NAMES[optimizer](learning_rate=learning_rate,
                                                            momentum=0.9)
        else:
            self.optimizer = OPTIMIZER_CLS_NAMES[optimizer](
                learning_rate=learning_rate)

        # Create a variable to track the global step
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # Gradient clipping
        if self.clip_gradients is not None:
            # Compute gradients
            trainable_vars = tf.trainable_variables()
            grads = tf.gradients(self.loss, trainable_vars)

            # TODO: Optionally add gradient noise

            if clip_gradients_by_norm:
                # Clip by norm
                self.clipped_grads = [tf.clip_by_norm(g,
                                                      clip_norm=self.clip_gradients) for g in grads]
            else:
                # Clip by absolute values
                self.clipped_grads = [tf.clip_by_value(g,
                                                       clip_value_min=-self.clip_gradients,
                                                       clip_value_max=self.clip_gradients) for g in grads]

            # TODO: Add histograms for variables, gradients (norms)
            self._tensorboard_statistics(trainable_vars)

            # Create gradient updates
            train_op = self.optimizer.apply_gradients(
                zip(self.clipped_grads, trainable_vars),
                global_step=global_step,
                name='train')

        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step
        train_op = self.optimizer.minimize(self.loss, global_step=global_step)

        return train_op

    def _add_scaled_noise_to_gradients(grads_and_vars, gradient_noise_scale):
        """Adds scaled noise from a 0-mean normal distribution to gradients."""
        raise NotImplementedError

    def decoder(self, decode_type, beam_width=None):
        """Operation for decoding.
        Args:
            decode_type: greedy or beam_search
            beam_width: beam width for beam search
        Return:
            decode_op: operation for decoding
        """
        if decode_type not in ['greedy', 'beam_search']:
            raise ValueError('decode_type is "greedy" or "beam_search".')

        if decode_type == 'greedy':
            decoded, _ = tf.nn.ctc_greedy_decoder(
                self.logits, tf.cast(self.seq_len_pl, tf.int32))

        elif decode_type == 'beam_search':
            if beam_width is None:
                raise ValueError('Set beam_width.')

            decoded, _ = tf.nn.ctc_beam_search_decoder(self.logits, tf.cast(self.seq_len_pl, tf.int32),
                                                       beam_width=beam_width)

        decode_op = tf.to_int32(decoded[0])
        return decode_op

    def posteriors(self, decode_op):
        """Operation for computing posteriors of each time steps.
        Args:
            decode_op: operation for decoding
        Return:
            posteriors_op: operation for computing posteriors for each class
        """
        # logits_3d : (max_timesteps, batch_size, num_classes)
        logits_2d = tf.reshape(self.logits, [-1, self.num_classes])
        posteriors_op = tf.nn.softmax(logits_2d)

        return posteriors_op

    def ler(self, decode_op):
        """Operation for computing LER.
        Args:
            decode_op: operation for decoding
        Return:
            ler_op: operation for computing label error rate
        """
        # Compute label error rate (normalize by label length)
        ler_op = tf.reduce_mean(tf.edit_distance(
            decode_op, self.labels_pl, normalize=True))
        # TODO: ここでの編集距離はラベルだから，文字に変換しないと正しいCERは得られない

        # Add a scalar summary for the snapshot of ler
        with tf.name_scope("ler"):
            self.summaries_train.append(tf.summary.scalar(
                'ler_train', ler_op))
            self.summaries_dev.append(tf.summary.scalar(
                'ler_dev', ler_op))

        return ler_op

    def _tensorboard_statistics(self, trainable_vars):
        """Compute statistics for TensorBoard plot.
        Args:
            trainable_vars:
        """
        # Histogram
        with tf.name_scope("train"):
            for var in trainable_vars:
                self.summaries_train.append(
                    tf.summary.histogram(var.name, var))
        with tf.name_scope("dev"):
            for var in trainable_vars:
                self.summaries_dev.append(
                    tf.summary.histogram(var.name, var))

        # Mean
        with tf.name_scope("mean_train"):
            for var in trainable_vars:
                self.summaries_train.append(tf.summary.scalar(var.name,
                                                              tf.reduce_mean(var)))
        with tf.name_scope("mean_dev"):
            for var in trainable_vars:
                self.summaries_dev.append(tf.summary.scalar(var.name,
                                                            tf.reduce_mean(var)))

        # Standard deviation
        with tf.name_scope("stddev_train"):
            for var in trainable_vars:
                self.summaries_train.append(tf.summary.scalar(var.name,
                                                              tf.sqrt(tf.reduce_mean(tf.square(var - tf.reduce_mean(var))))))
        with tf.name_scope("stddev_dev"):
            for var in trainable_vars:
                self.summaries_dev.append(tf.summary.scalar(var.name,
                                                            tf.sqrt(tf.reduce_mean(tf.square(var - tf.reduce_mean(var))))))

        # Max
        with tf.name_scope("max_train"):
            for var in trainable_vars:
                self.summaries_train.append(tf.summary.scalar(var.name,
                                                              tf.reduce_max(var)))
        with tf.name_scope("max_dev"):
            for var in trainable_vars:
                self.summaries_dev.append(
                    tf.summary.scalar(var.name, tf.reduce_max(var)))

        # Min
        with tf.name_scope("min_train"):
            for var in trainable_vars:
                self.summaries_train.append(tf.summary.scalar(var.name,
                                                              tf.reduce_min(var)))
        with tf.name_scope("min_dev"):
            for var in trainable_vars:
                self.summaries_dev.append(tf.summary.scalar(var.name,
                                                            tf.reduce_min(var)))
