#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class of the multi-task CTC model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from models.ctc.base import CTCBase


class multitaskCTCBase(CTCBase):
    """Hierarchical Connectionist Temporal Classification (CTC) network.
    Args:
        input_size (int): the dimensions of input vectors
        splice (int): frames to splice. Default is 1 frame.
        num_classes_main (int): the number of classes of target labels in the
            main task (except for a blank label)
        num_classes_sub (int): the number of classes of target labels in the
            sub task (except for a blank label)
        main_task_weight (float): the weight of loss of the main task.
            Set between 0 to 1.
        clip_grad (float): range of gradient clipping (> 0)
        weight_decay (float): a parameter for weight decay
    """

    def __init__(self, input_size, splice, num_classes_main, num_classes_sub,
                 main_task_weight, clip_grad, weight_decay):

        CTCBase.__init__(self, input_size, splice, num_classes_main,
                         clip_grad, weight_decay)

        self.num_classes_sub = int(num_classes_sub) + 1  # plus blank label
        if float(main_task_weight) < 0 or float(main_task_weight) > 1:
            raise ValueError('Set main_task_weight between 0 to 1.')
        self.main_task_weight = float(main_task_weight)
        self.sub_task_weight = 1 - self.main_task_weight

        # Placeholder for multi-task
        self.labels_sub_pl_list = []

    def __call__(self, inputs, inputs_seq_len, keep_prob_input,
                 keep_prob_hidden, keep_prob_output):
        """Construct model graph.
        Args:
            inputs: A tensor of size `[B, T, input_size]`
            inputs_seq_len: A tensor of size `[B]`
            keep_prob_input: A float value. A probability to keep nodes in
                the input-hidden connection
            keep_prob_hidden: A float value. A probability to keep nodes in
                the hidden-hidden connection
            keep_prob_output: A float value. A probability to keep nodes in
                the hidden-output connection
        Returns:
            logits_main: A tensor of size `[T, B, num_classes]`
            logits_sub: A tensor of size `[T, B, num_classes]`
        """
        logits_main, logits_sub, final_state, final_state_sub = self.encoder(
            inputs, inputs_seq_len, keep_prob_input,
            keep_prob_hidden, keep_prob_output)

        return logits_main, logits_sub

    def create_placeholders(self):
        """Create placeholders and append them to list."""
        self.inputs_pl_list.append(
            tf.placeholder(tf.float32, shape=[None, None, self.input_size],
                           name='input'))
        self.labels_pl_list.append(
            tf.SparseTensor(tf.placeholder(tf.int64, name='indices'),
                            tf.placeholder(tf.int32, name='values'),
                            tf.placeholder(tf.int64, name='shape')))
        self.labels_sub_pl_list.append(
            tf.SparseTensor(tf.placeholder(tf.int64, name='indices_sub'),
                            tf.placeholder(tf.int32, name='values_sub'),
                            tf.placeholder(tf.int64, name='shape_sub')))
        self.inputs_seq_len_pl_list.append(
            tf.placeholder(tf.int64, shape=[None], name='inputs_seq_len'))
        self.keep_prob_input_pl_list.append(
            tf.placeholder(tf.float32, name='keep_prob_input'))
        self.keep_prob_hidden_pl_list.append(
            tf.placeholder(tf.float32, name='keep_prob_hidden'))
        self.keep_prob_output_pl_list.append(
            tf.placeholder(tf.float32, name='keep_prob_output'))

    def compute_loss(self, inputs, labels_main, labels_sub, inputs_seq_len,
                     keep_prob_input, keep_prob_hidden, keep_prob_output,
                     scope=None):
        """Operation for computing ctc loss.
        Args:
            inputs: A tensor of size `[B, T, input_size]`
            labels_main: A SparseTensor of target labels in the main task
            labels_sub: A SparseTensor of target labels in the sub task
            inputs_seq_len: A tensor of size `[B]`
            keep_prob_input: A float value. A probability to keep nodes in
                the input-hidden layer
            keep_prob_hidden: A float value. A probability to keep nodes in
                the hidden-hidden layers
            keep_prob_output: A float value. A probability to keep nodes in
                the hidden-output layer
            scope: A scope in the model tower
        Returns:
            total_loss: operation for computing total ctc loss
            logits_main: A tensor of size `[T, B, input_size]`
            logits_sub: A tensor of size `[T, B, input_size]`
        """
        # Build model graph
        logits_main, logits_sub = self(
            inputs, inputs_seq_len,
            keep_prob_input, keep_prob_hidden, keep_prob_output)

        # Weight decay
        if self.weight_decay > 0:
            with tf.name_scope("weight_decay_loss"):
                weight_sum = 0
                for var in tf.trainable_variables():
                    if 'bias' not in var.name.lower():
                        weight_sum += tf.nn.l2_loss(var)
                tf.add_to_collection('losses', weight_sum * self.weight_decay)

        with tf.name_scope("ctc_loss_main"):
            ctc_losses = tf.nn.ctc_loss(
                labels_main,
                logits_main,
                tf.cast(inputs_seq_len, tf.int32),
                preprocess_collapse_repeated=False,
                ctc_merge_repeated=True,
                ignore_longer_outputs_than_inputs=False,
                time_major=True)
            ctc_loss_main = tf.reduce_mean(
                ctc_losses, name='ctc_loss_mean_main')
            tf.add_to_collection(
                'losses', ctc_loss_main * self.main_task_weight)

        with tf.name_scope("ctc_loss_sub"):
            ctc_losses = tf.nn.ctc_loss(
                labels_sub,
                logits_sub,
                tf.cast(inputs_seq_len, tf.int32),
                preprocess_collapse_repeated=False,
                ctc_merge_repeated=True,
                ignore_longer_outputs_than_inputs=False,
                time_major=True)
            ctc_loss_sub = tf.reduce_mean(
                ctc_losses, name='ctc_loss_mean_sub')
            tf.add_to_collection(
                'losses', ctc_loss_sub * self.sub_task_weight)

        # Compute total loss
        total_loss = tf.add_n(tf.get_collection('losses', scope),
                              name='total_loss')

        # Add a scalar summary for the snapshot of loss
        if self.weight_decay > 0:
            self.summaries_train.append(
                tf.summary.scalar('weight_loss_train',
                                  weight_sum * self.weight_decay))
            self.summaries_dev.append(
                tf.summary.scalar('weight_loss_dev',
                                  weight_sum * self.weight_decay))
            self.summaries_train.append(
                tf.summary.scalar('total_loss_train', total_loss))
            self.summaries_dev.append(
                tf.summary.scalar('total_loss_dev', total_loss))

        self.summaries_train.append(
            tf.summary.scalar('ctc_loss_main_train',
                              ctc_loss_main * self.main_task_weight))
        self.summaries_dev.append(
            tf.summary.scalar('ctc_loss_main_dev',
                              ctc_loss_main * self.main_task_weight))

        self.summaries_train.append(
            tf.summary.scalar('ctc_loss_sub_train',
                              ctc_loss_sub * self.sub_task_weight))
        self.summaries_dev.append(
            tf.summary.scalar('ctc_loss_sub_dev',
                              ctc_loss_sub * self.sub_task_weight))

        return total_loss, logits_main, logits_sub

    def decoder(self, logits_main, logits_sub, inputs_seq_len, decode_type,
                beam_width=None):
        """Operation for decoding.
        Args:
            logits_main: A tensor of size `[T, B, input_size]`
            logits_sub: A tensor of size `[T, B, input_size]`
            inputs_seq_len: A tensor of size `[B]`
            decode_type: greedy or beam_search
            beam_width: beam width for beam search
        Return:
            decode_op_main: operation for decoding of the main task
            decode_op_sub: operation for decoding of the sub task
        """
        if decode_type not in ['greedy', 'beam_search']:
            raise ValueError('decode_type is "greedy" or "beam_search".')

        if decode_type == 'greedy':
            decoded_main, _ = tf.nn.ctc_greedy_decoder(
                logits_main, tf.cast(inputs_seq_len, tf.int32))
            decoded_sub, _ = tf.nn.ctc_greedy_decoder(
                logits_sub, tf.cast(inputs_seq_len, tf.int32))

        elif decode_type == 'beam_search':
            if beam_width is None:
                raise ValueError('Set beam_width.')

            decoded_main, _ = tf.nn.ctc_beam_search_decoder(
                logits_main, tf.cast(inputs_seq_len, tf.int32),
                beam_width=beam_width)
            decoded_sub, _ = tf.nn.ctc_beam_search_decoder(
                logits_sub, tf.cast(inputs_seq_len, tf.int32),
                beam_width=beam_width)

        decode_op_main = tf.to_int32(decoded_main[0])
        decode_op_sub = tf.to_int32(decoded_sub[0])

        return decode_op_main, decode_op_sub

    def posteriors(self, logits_main, logits_sub):
        """Operation for computing posteriors of each time steps.
        Args:
            logits_main: A tensor of size `[T, B, input_size]`
            logits_sub: A tensor of size `[T, B, input_size]`
        Return:
            posteriors_op_main: operation for computing posteriors for each
                class in the main task
            posteriors_op_sub: operation for computing posteriors for each
                class in the sub task
        """
        # Convert to batch-major: `[batch_size, max_time, num_classes]'
        logits_main = tf.transpose(logits_main, (1, 0, 2))
        logits_sub = tf.transpose(logits_sub, (1, 0, 2))

        logits_2d_main = tf.reshape(logits_main,
                                    shape=[-1, self.num_classes])
        posteriors_op_main = tf.nn.softmax(logits_2d_main)

        logits_2d_sub = tf.reshape(logits_sub,
                                   shape=[-1, self.num_classes_sub])
        posteriors_op_sub = tf.nn.softmax(logits_2d_sub)

        return posteriors_op_main, posteriors_op_sub

    def compute_ler(self, decode_op_main, decode_op_sub,
                    labels_main, labels_sub):
        """Operation for computing LER (Label Error Rate).
        Args:
            decode_op_main: operation for decoding of the main task
            decode_op_sub: operation for decoding of the sub task
            labels_main: A SparseTensor of target labels in the main task
            labels_sub: A SparseTensor of target labels in the sub task
        Return:
            ler_op_main: operation for computing LER of the main task
            ler_op_sub: operation for computing LER of the sub task
        """
        # Compute LER (normalize by label length)
        ler_op_main = tf.reduce_mean(tf.edit_distance(
            decode_op_main, labels_main, normalize=True))
        ler_op_sub = tf.reduce_mean(tf.edit_distance(
            decode_op_sub, labels_sub, normalize=True))

        # Add a scalar summary for the snapshot of LER
        self.summaries_train.append(tf.summary.scalar(
            'ler_main_train', ler_op_main))
        self.summaries_train.append(tf.summary.scalar(
            'ler_sub_train', ler_op_sub))
        self.summaries_dev.append(tf.summary.scalar(
            'ler_main_dev', ler_op_main))
        self.summaries_dev.append(tf.summary.scalar(
            'ler_sub_dev', ler_op_sub))

        return ler_op_main, ler_op_sub
