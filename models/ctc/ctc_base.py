#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""CTC network."""

import tensorflow as tf


class ctcBase(object):
    """Connectionist Temporal Classification (CTC) network.
    Args:
        batch_size: int, batch size of mini batch
        input_size: int, the dimension of input vectors
        num_cell: int, the number of memory cells in each layer
        num_layers: int, the number of layers
        output_size: int, the number of nodes in softmax layer (except for blank class)
        parameter_init: A float value. Range of uniform distribution to initialize weight parameters
        clip_grad: A float value. Range of gradient clipping (non-negative)
        clip_activation: A float value. Range of activation clipping (non-negative)
        dropout_ratio_input: A float value. Dropout ratio in input-hidden layers
        dropout_ratio_hidden: A float value. Dropout ratio in hidden-hidden layers
    """

    def __init__(self,
                 batch_size,
                 input_size,
                 num_cell,
                 num_layers,
                 output_size,
                 parameter_init,
                 clip_grad,
                 clip_activation,
                 dropout_ratio_input,
                 dropout_ratio_hidden):

        # network size
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_cell = num_cell
        self.num_layers = num_layers
        self.num_classes = output_size + 1  # plus blank label

        # network settings
        self.parameter_init = parameter_init
        self.clip_grad = clip_grad
        self.clip_activation = clip_activation

        # dropout
        if dropout_ratio_input == 1.0 and dropout_ratio_hidden == 1.0:
            self.dropout = False
        else:
            self.dropout = True
        self.dropout_ratio_input = dropout_ratio_input
        self.dropout_ratio_hidden = dropout_ratio_hidden

        # summaries for TensorBoard
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
        # self.seq_len_pl = tf.placeholder(tf.int32, shape=[None], name='seq_len')
        self.seq_len_pl = tf.placeholder(tf.int64,
                                         shape=[None],
                                         name='seq_len')

        # for dropout
        self.keep_prob_input_pl = tf.placeholder(tf.float32,
                                                 name='keep_prob_input')
        self.keep_prob_hidden_pl = tf.placeholder(tf.float32,
                                                  name='keep_prob_hidden')

        # learning rate
        self.lr_pl = tf.placeholder(tf.float32, name='learning_rate')

    def loss(self):
        """Operation for computing ctc loss.
        Returns:
            loss: operation for computing ctc loss
        """
        with tf.name_scope("ctc_loss"):
            loss = tf.nn.ctc_loss(
                self.labels_pl, self.logits, tf.cast(self.seq_len_pl, tf.int32))
            self.loss = tf.reduce_mean(loss, name='ctc_loss_mean')

            # Add a scalar summary for the snapshot of loss
            self.summaries_train.append(
                tf.summary.scalar('loss (train)', self.loss))
            self.summaries_dev.append(
                tf.summary.scalar('loss (dev)', self.loss))

            return self.loss

    def train(self, optimizer, learning_rate_init=None, is_scheduled=False):
        """Operation for training.
        Args:
            optimizer: adam or adadelta or rmsprop or sgd or momentum
            learning_rate_init: initial learning rate
            is_scheduled: if True, schedule learning rate at each epoch
        Returns:
            train_op: operation for training
        """
        if optimizer not in ['adam', 'adadelta', 'rmsprop', 'sgd', 'momentum']:
            raise ValueError(
                'Optimizer is "adam" "adadelta" or "rmsprop" or "sgd" or "momentum".')

        # select parameter update method
        if is_scheduled:
            learning_rate = self.lr_pl
        else:
            learning_rate = learning_rate_init
        if optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif optimizer == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate)
        elif optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                                   momentum=0.9)
        self.optimizer = optimizer

        # Create a variable to track the global step
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # gradient clipping
        if self.clip_grad is not None:
            tvars = tf.trainable_variables()
            grads = tf.gradients(self.loss, tvars)

            # clip by absolute values
            self.clipped_grads = [tf.clip_by_value(g,
                                                   clip_value_min=-self.clip_grad,
                                                   clip_value_max=self.clip_grad) for g in grads]
            # clip by norm
            # self.clipped_grads = [tf.clip_by_norm(g, clip_norm=self.clip_grad) for g in grads]

            train_op = optimizer.apply_gradients(
                zip(self.clipped_grads, tvars), global_step=global_step)

        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step
        train_op = optimizer.minimize(self.loss, global_step=global_step)

        return train_op

    def greedy_decoder(self):
        """Operation for greedy decoding.
        Return:
            decode_op: operation for decoding
        """
        decoded, _ = tf.nn.ctc_greedy_decoder(
            self.logits, tf.cast(self.seq_len_pl, tf.int32))
        decode_op = tf.to_int32(decoded[0])
        return decode_op

    def beam_search_decoder(self, beam_width):
        """Operation for beam search decoding.
        Args:
            beam_width: beam width for beam search
        Return:
            decode_op: operation for decoding
        """
        # decode
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
        # compute phone error rate (normalize by label length)
        ler_op = tf.reduce_mean(tf.edit_distance(
            decode_op, self.labels_pl, normalize=True))
        # TODO: ここでの編集距離は数字だから，文字に変換しないと正しい結果は得られない

        # add a scalar summary for the snapshot of ler
        self.summaries_train.append(tf.summary.scalar(
            'LER (train)', ler_op))
        self.summaries_dev.append(tf.summary.scalar(
            'LER (dev)', ler_op))
        return ler_op
