#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Multi-task Bidirectional LSTM-CTC model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from models.ctc.ctc_base import ctcBase


class Multitask_BLSTM_CTC(ctcBase):
    """Multi-task Bidirectional LSTM-CTC model.
    Args:
        batch_size: int, batch size of mini batch
        input_size: int, the dimensions of input vectors
        num_unit: int, the number of units in each layer
        num_layer_main: int, the number of layers of the main task
        num_layer_second: int, the number of layers of the second task. Set
            between 1 to num_layer_main
        num_classes_main: int, the number of classes of target labels in the
            main task (except for a blank label)
        num_classes_second: int, the number of classes of target labels in the
            second task (except for a blank label)
        main_task_weight: A float value. The weight of loss of the main task.
            Set between 0 to 1
        parameter_init: A float value. Range of uniform distribution to
            initialize weight parameters
        clip_grad: A float value. Range of gradient clipping (> 0)
        clip_activation: A float value. Range of activation clipping (> 0)
        dropout_ratio_input: A float value. Dropout ratio in input-hidden
            layers
        dropout_ratio_hidden: A float value. Dropout ratio in hidden-hidden
            layers
        num_proj: int, the number of nodes in recurrent projection layer
        weight_decay: A float value. Regularization parameter for weight decay
        bottleneck_dim: int, the dimensions of the bottleneck layer
    """

    def __init__(self,
                 batch_size,
                 input_size,
                 num_unit,
                 num_layer_main,
                 num_layer_second,
                 num_classes_main,
                 num_classes_second,
                 main_task_weight,
                 parameter_init=0.1,
                 clip_grad=None,
                 clip_activation=None,
                 dropout_ratio_input=1.0,
                 dropout_ratio_hidden=1.0,
                 num_proj=None,
                 weight_decay=0.0,
                 bottleneck_dim=None,
                 name='multitask_blstm_ctc'):

        ctcBase.__init__(self, batch_size, input_size, num_unit,
                         num_layer_main, num_classes_main, parameter_init,
                         clip_grad, clip_activation,
                         dropout_ratio_input, dropout_ratio_hidden,
                         weight_decay, name)

        self.num_proj = None if num_proj == 0 else num_proj
        self.bottleneck_dim = bottleneck_dim

        if num_layer_second < 1 or num_layer_second > num_layer_main:
            raise ValueError(
                'Set num_layer_second between 1 to num_layer_main.')
        self.num_layer_second = num_layer_second
        self.num_classes_second = num_classes_second + 1  # plus blank label

        if main_task_weight < 0 or main_task_weight > 1:
            raise ValueError('Set main_task_weight between 0 to 1.')
        self.main_task_weight = main_task_weight
        self.second_task_weight = 1 - main_task_weight

    def _build(self, inputs, inputs_seq_len, keep_prob_input,
               keep_prob_hidden):
        """Construct model graph.
        Args:
            inputs: A tensor of `[batch_size, max_time, input_dim]`
            inputs_seq_len: A tensor of `[batch_size]`
            keep_prob_input:
            keep_prob_hidden:
        Returns:
            logits: A tensor of size `[max_time, batch_size, input_size]`
        """
        # Dropout for inputs
        outputs = tf.nn.dropout(inputs,
                                keep_prob_input,
                                name='dropout_input')

        # inputs: `[batch_size, max_time, input_size_splice]`
        batch_size = tf.shape(inputs)[0]

        # Hidden layers
        for i_layer in range(self.num_layer):
            with tf.name_scope('blstm_hidden' + str(i_layer + 1)):

                initializer = tf.random_uniform_initializer(
                    minval=-self.parameter_init,
                    maxval=self.parameter_init)

                lstm_fw = tf.contrib.rnn.LSTMCell(
                    self.num_unit,
                    use_peepholes=True,
                    cell_clip=self.clip_activation,
                    initializer=initializer,
                    num_proj=self.num_proj,
                    forget_bias=1.0,
                    state_is_tuple=True)
                lstm_bw = tf.contrib.rnn.LSTMCell(
                    self.num_unit,
                    use_peepholes=True,
                    cell_clip=self.clip_activation,
                    initializer=initializer,
                    num_proj=self.num_proj,
                    forget_bias=1.0,
                    state_is_tuple=True)

                # Dropout for outputs of each layer
                lstm_fw = tf.contrib.rnn.DropoutWrapper(
                    lstm_fw,
                    output_keep_prob=keep_prob_hidden)
                lstm_bw = tf.contrib.rnn.DropoutWrapper(
                    lstm_bw,
                    output_keep_prob=keep_prob_hidden)

                # _init_state_fw = lstm_fw.zero_state(self.batch_size,
                #                                     tf.float32)
                # _init_state_bw = lstm_bw.zero_state(self.batch_size,
                #                                     tf.float32)
                # initial_state_fw=_init_state_fw,
                # initial_state_bw=_init_state_bw,

                # Ignore 2nd return (the last state)
                (outputs_fw, outputs_bw), final_state = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=lstm_fw,
                    cell_bw=lstm_bw,
                    inputs=outputs,
                    sequence_length=inputs_seq_len,
                    dtype=tf.float32,
                    scope='blstm_dynamic' + str(i_layer + 1))

                outputs = tf.concat(axis=2, values=[outputs_fw, outputs_bw])

                if i_layer == self.num_layer_second - 1:
                    # Reshape to apply the same weights over the timesteps
                    if self.num_proj is None:
                        output_node = self.num_unit * 2
                    else:
                        output_node = self.num_proj * 2
                    outputs_hidden = tf.reshape(
                        outputs, shape=[-1, output_node])

                    with tf.name_scope('output_second'):
                        # Affine
                        W_output_second = tf.Variable(tf.truncated_normal(
                            shape=[output_node, self.num_classes_second],
                            stddev=0.1, name='W_output_second'))
                        b_output_second = tf.Variable(tf.zeros(
                            shape=[self.num_classes_second],
                            name='b_output_second'))
                        logits_second_2d = tf.matmul(
                            outputs_hidden, W_output_second) + b_output_second

                        # Reshape back to the original shape
                        logits_second = tf.reshape(
                            logits_second_2d,
                            shape=[batch_size, -1, self.num_classes_second])

                        # Convert to time-major:
                        # `[max_time, batch_size, num_classes]`
                        logits_second = tf.transpose(logits_second, (1, 0, 2))

        # Reshape to apply the same weights over the timesteps
        if self.num_proj is None:
            output_node = self.num_unit * 2
        else:
            output_node = self.num_proj * 2
        outputs = tf.reshape(outputs, shape=[-1, output_node])

        if self.bottleneck_dim is not None and self.bottleneck_dim != 0:
            with tf.name_scope('bottleneck'):
                # Affine
                W_bottleneck = tf.Variable(tf.truncated_normal(
                    shape=[output_node, self.bottleneck_dim],
                    stddev=0.1, name='W_bottleneck'))
                b_bottleneck = tf.Variable(tf.zeros(
                    shape=[self.bottleneck_dim], name='b_bottleneck'))
                outputs = tf.matmul(outputs, W_bottleneck) + b_bottleneck
                output_node = self.bottleneck_dim

        with tf.name_scope('output_main'):
            # Affine
            W_output_main = tf.Variable(tf.truncated_normal(
                shape=[output_node, self.num_classes],
                stddev=0.1, name='W_output_main'))
            b_output_main = tf.Variable(tf.zeros(
                shape=[self.num_classes], name='b_output_main'))
            logits_main_2d = tf.matmul(outputs, W_output_main) + b_output_main

            # Reshape back to the original shape
            logits_main = tf.reshape(
                logits_main_2d, shape=[batch_size, -1, self.num_classes])

            # Convert to time-major: `[max_time, batch_size, num_classes]'
            logits_main = tf.transpose(logits_main, (1, 0, 2))

            return logits_main, logits_second

    def compute_loss(self, inputs, labels_main, labels_second, inputs_seq_len,
                     keep_prob_input, keep_prob_hidden, num_gpu=1, scope=None):
        """Operation for computing ctc loss.
        Args:
            inputs: A tensor of size `[batch_size, max_time, input_size]`
            labels_main: A SparseTensor of target labels in the main task
            labels_second: A SparseTensor of target labels in the second task
            inputs_seq_len: A tensor of size `[batch_size]`
            keep_prob_input: A float value. A probability to keep nodes in
                input-hidden layers
            keep_prob_hidden: A float value. A probability to keep nodes in
                hidden-hidden layers
            num_gpu: the number of GPUs
        Returns:
            loss: operation for computing ctc loss
            logits_main: A tensor of size `[max_time, batch_size, input_size]`
            logits_second: A tensor of size `[max_time, batch_size, input_size]`
        """
        # Build model graph
        logits_main, logits_second = self._build(
            inputs, inputs_seq_len, keep_prob_input, keep_prob_hidden)

        # Weight decay
        with tf.name_scope("weight_decay_loss"):
            weight_sum = 0
            for var in tf.trainable_variables():
                if 'bias' not in var.name.lower():
                    weight_sum += tf.nn.l2_loss(var)
            tf.add_to_collection('losses', weight_sum * self.weight_decay)

        with tf.name_scope("ctc_loss_main"):
            ctc_losses = tf.nn.ctc_loss(labels_main,
                                        logits_main,
                                        tf.cast(inputs_seq_len, tf.int32),
                                        preprocess_collapse_repeated=False,
                                        ctc_merge_repeated=True,
                                        ignore_longer_outputs_than_inputs=False,
                                        time_major=True)
            ctc_loss = tf.reduce_mean(
                ctc_losses, name='ctc_loss_main')
            tf.add_to_collection(
                'losses', ctc_loss * self.main_task_weight)

            self.summaries_train.append(
                tf.summary.scalar('ctc_loss_main_train',
                                  ctc_loss * self.main_task_weight))
            self.summaries_dev.append(
                tf.summary.scalar('ctc_loss_main_dev',
                                  ctc_loss * self.main_task_weight))

        with tf.name_scope("ctc_loss_second"):
            ctc_losses = tf.nn.ctc_loss(labels_second,
                                        logits_second,
                                        tf.cast(inputs_seq_len, tf.int32),
                                        preprocess_collapse_repeated=False,
                                        ctc_merge_repeated=True,
                                        ignore_longer_outputs_than_inputs=False,
                                        time_major=True)
            ctc_loss = tf.reduce_mean(
                ctc_losses, name='ctc_loss_second')
            tf.add_to_collection(
                'losses', ctc_loss * self.second_task_weight)

            self.summaries_train.append(
                tf.summary.scalar('ctc_loss_second_train',
                                  ctc_loss * self.second_task_weight))
            self.summaries_dev.append(
                tf.summary.scalar('ctc_loss_second_dev',
                                  ctc_loss * self.second_task_weight))

        # Compute total loss
        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

        # Add a scalar summary for the snapshot of loss
        with tf.name_scope("total_loss"):
            self.summaries_train.append(
                tf.summary.scalar('total_loss_train', loss))
            self.summaries_dev.append(
                tf.summary.scalar('total_loss_dev', loss))

        return loss, logits_main, logits_second

    def decoder(self, logits_main, logits_second, inputs_seq_len, decode_type,
                beam_width=None):
        """Operation for decoding.
        Args:
            logits_main: A tensor of size `[max_time, batch_size, input_size]`
            logits_second: A tensor of size `[max_time, batch_size, input_size]`
            inputs_seq_len: A tensor of size `[batch_size]`
            decode_type: greedy or beam_search
            beam_width: beam width for beam search
        Return:
            decode_op_main: operation for decoding of the main task
            decode_op_second: operation for decoding of the second task
        """
        if decode_type not in ['greedy', 'beam_search']:
            raise ValueError('decode_type is "greedy" or "beam_search".')

        if decode_type == 'greedy':
            decoded_main, _ = tf.nn.ctc_greedy_decoder(
                logits_main, tf.cast(inputs_seq_len, tf.int32))
            decoded_second, _ = tf.nn.ctc_greedy_decoder(
                logits_second, tf.cast(inputs_seq_len, tf.int32))

        elif decode_type == 'beam_search':
            if beam_width is None:
                raise ValueError('Set beam_width.')

            decoded_main, _ = tf.nn.ctc_beam_search_decoder(
                logits_main, tf.cast(inputs_seq_len, tf.int32),
                beam_width=beam_width)
            decoded_second, _ = tf.nn.ctc_beam_search_decoder(
                logits_second, tf.cast(inputs_seq_len, tf.int32),
                beam_width=beam_width)

        decode_op_main = tf.to_int32(decoded_main[0])
        decode_op_second = tf.to_int32(decoded_second[0])

        return decode_op_main, decode_op_second

    def posteriors(self, logits_main, logits_second):
        """Operation for computing posteriors of each time steps.
        Args:
            logits_main: A tensor of size `[max_time, batch_size, input_size]`
            logits_second: A tensor of size `[max_time, batch_size, input_size]`
        Return:
            posteriors_op_main: operation for computing posteriors for each
                class in the main task
            posteriors_op_second: operation for computing posteriors for each
                class in the second task
        """
        # Convert to batch-major: `[batch_size, max_time, num_classes]'
        logits_main = tf.transpose(logits_main, (1, 0, 2))
        logits_second = tf.transpose(logits_second, (1, 0, 2))

        logits_2d_main = tf.reshape(logits_main,
                                    shape=[-1, self.num_classes])
        posteriors_op_main = tf.nn.softmax(logits_2d_main)

        logits_2d_second = tf.reshape(logits_second,
                                      shape=[-1, self.num_classes_second])
        posteriors_op_second = tf.nn.softmax(logits_2d_second)

        return posteriors_op_main, posteriors_op_second

    def compute_ler(self, decode_op_main, decode_op_second,
                    labels_main, labels_second):
        """Operation for computing LER (Label Error Rate).
        Args:
            decode_op_main: operation for decoding of the main task
            decode_op_second: operation for decoding of the second task
            labels_main: A SparseTensor of target labels in the main task
            labels_second: A SparseTensor of target labels in the second task
        Return:
            ler_op_main: operation for computing LER of the main task
            ler_op_second: operation for computing LER of the second task
        """
        # Compute LER (normalize by label length)
        ler_op_main = tf.reduce_mean(tf.edit_distance(
            decode_op_main, labels_main, normalize=True))
        ler_op_second = tf.reduce_mean(tf.edit_distance(
            decode_op_second, labels_second, normalize=True))
        # NOTE: ここでの編集距離はラベルだから，文字に変換しないと正しいCERは得られない

        # Add a scalar summary for the snapshot of LER
        with tf.name_scope("ler"):
            self.summaries_train.append(tf.summary.scalar(
                'ler_main_train', ler_op_main))
            self.summaries_train.append(tf.summary.scalar(
                'ler_second_train', ler_op_second))
            self.summaries_dev.append(tf.summary.scalar(
                'ler_main_dev', ler_op_main))
            self.summaries_dev.append(tf.summary.scalar(
                'ler_second_dev', ler_op_second))

        return ler_op_main, ler_op_second
