#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Multi-task Bidirectional LSTM-CTC model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .ctc_base import ctcBase


class Multitask_BLSTM_CTC(ctcBase):
    """Multi-task Bidirectional LSTM-CTC model.
    Args:
        batch_size: int, batch size of mini batch
        input_size: int, the dimensions of input vectors
        num_unit: int, the number of units in each layer
        num_layer_main: int, the number of layers of the main task
        num_layer_second: int, the number of layers of the second task. Set
            between 1 to num_layer_main
        output_size_main: int, the number of nodes in softmax layer of the main
            task (except for blank class)
        output_size_second: int, the number of nodes in softmax layer of the
            second task (except for blank class)
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
        bottleneck_dim: not used
    """

    def __init__(self,
                 batch_size,
                 input_size,
                 num_unit,
                 num_layer_main,
                 num_layer_second,
                 output_size_main,
                 output_size_second,
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
                         num_layer_main, output_size_main, parameter_init,
                         clip_grad, clip_activation,
                         dropout_ratio_input, dropout_ratio_hidden,
                         weight_decay, name)

        self.num_proj = None if num_proj == 0 else num_proj
        # TODO: implement projection layer
        # TODO: implement bottleneck layer

        if num_layer_second < 1 or num_layer_second > num_layer_main:
            raise ValueError(
                'Set num_layer_second between 1 to num_layer_main.')
        self.num_layer_second = num_layer_second
        self.num_classes_second = output_size_second + 1  # plus blank label

        if main_task_weight < 0 or main_task_weight > 1:
            raise ValueError('Set main_task_weight between 0 to 1.')
        self.main_task_weight = main_task_weight
        self.second_task_weight = 1 - main_task_weight

    def define(self):
        """Construct model graph."""
        # Generate placeholders
        self._generate_placeholer()
        self.label_indices_second = tf.placeholder(tf.int64,
                                                   name='indices_second')
        self.label_values_second = tf.placeholder(tf.int32,
                                                  name='values_second')
        self.label_shape_second = tf.placeholder(tf.int64,
                                                 name='shape_second')
        self.labels_second = tf.SparseTensor(self.label_indices_second,
                                             self.label_values_second,
                                             self.label_shape_second)

        # Dropout for Input
        outputs = tf.nn.dropout(self.inputs,
                                self.keep_prob_input,
                                name='dropout_input')

        # `[batch_size, max_time, input_size_splice]`
        batch_size = tf.shape(self.inputs)[0]

        # Hidden layers
        for i_layer in range(self.num_layer):
            with tf.name_scope('BiLSTM_hidden' + str(i_layer + 1)):

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

                # Dropout (output)
                lstm_fw = tf.contrib.rnn.DropoutWrapper(
                    lstm_fw,
                    output_keep_prob=self.keep_prob_hidden)
                lstm_bw = tf.contrib.rnn.DropoutWrapper(
                    lstm_bw,
                    output_keep_prob=self.keep_prob_hidden)

                # _init_state_fw = lstm_fw.zero_state(self.batch_size,
                #                                     tf.float32)
                # _init_state_bw = lstm_bw.zero_state(self.batch_size,
                #                                     tf.float32)
                # initial_state_fw=_init_state_fw,
                # initial_state_bw=_init_state_bw,

                # Ignore 2nd return (the last state)
                (outputs_fw, outputs_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=lstm_fw,
                    cell_bw=lstm_bw,
                    inputs=outputs,
                    sequence_length=self.seq_len,
                    dtype=tf.float32,
                    scope='BiLSTM_' + str(i_layer + 1))

                outputs = tf.concat(axis=2, values=[outputs_fw, outputs_bw])

                if i_layer == self.num_layer_second:
                    # Reshape to apply the same weights over the timesteps
                    if self.num_proj is None:
                        output_node = self.num_unit * 2
                    else:
                        output_node = self.num_proj * 2
                    outputs_hidden = tf.reshape(
                        outputs, shape=[-1, output_node])

                    with tf.name_scope('output_second'):
                        # Affine
                        W_output = tf.Variable(tf.truncated_normal(
                            shape=[output_node, self.num_classes_second],
                            stddev=0.1, name='W_output_second'))
                        b_output = tf.Variable(tf.zeros(
                            shape=[self.num_classes_second],
                            name='b_output_second'))
                        logits_2d = tf.matmul(
                            outputs_hidden, W_output) + b_output

                        # Reshape back to the original shape
                        logits_3d = tf.reshape(
                            logits_2d,
                            shape=[batch_size, -1, self.num_classes_second])

                        # Convert to `[max_time, batch_size, num_classes]`
                        self.logits_second = tf.transpose(logits_3d, (1, 0, 2))

        # Reshape to apply the same weights over the timesteps
        if self.num_proj is None:
            output_node = self.num_unit * 2
        else:
            output_node = self.num_proj * 2
        outputs = tf.reshape(outputs, shape=[-1, output_node])

        with tf.name_scope('output_main'):
            # Affine
            W_output = tf.Variable(tf.truncated_normal(
                shape=[output_node, self.num_classes],
                stddev=0.1, name='W_output_main'))
            b_output = tf.Variable(tf.zeros(
                shape=[self.num_classes], name='b_output_main'))
            logits_2d = tf.matmul(outputs, W_output) + b_output

            # Reshape back to the original shape
            logits_3d = tf.reshape(
                logits_2d, shape=[batch_size, -1, self.num_classes])

            # Convert to (max_time, batch_size, num_classes)
            self.logits_main = tf.transpose(logits_3d, (1, 0, 2))

    def compute_loss(self):
        """Operation for computing ctc loss.
        Returns:
            loss: operation for computing ctc loss
        """
        # Weight decay
        weight_sum = 0
        for var in tf.trainable_variables():
            if 'bias' not in var.name.lower():
                weight_sum += tf.nn.l2_loss(var)
        tf.add_to_collection('losses', weight_sum * self.weight_decay)

        with tf.name_scope("ctc_loss_main"):
            ctc_loss = tf.nn.ctc_loss(self.labels,
                                      self.logits_main,
                                      tf.cast(self.seq_len, tf.int32))
            ctc_loss_mean = tf.reduce_mean(
                ctc_loss, name='ctc_loss_main_mean')
            tf.add_to_collection(
                'losses', ctc_loss_mean * self.main_task_weight)

            self.summaries_train.append(
                tf.summary.scalar('ctc_loss_train_main',
                                  ctc_loss_mean * self.main_task_weight))
            self.summaries_dev.append(
                tf.summary.scalar('ctc_loss_dev_main',
                                  ctc_loss_mean * self.main_task_weight))

        with tf.name_scope("ctc_loss_second"):
            ctc_loss = tf.nn.ctc_loss(self.labels_second,
                                      self.logits_second,
                                      tf.cast(self.seq_len, tf.int32))
            ctc_loss_mean = tf.reduce_mean(
                ctc_loss, name='ctc_loss_second_mean')
            tf.add_to_collection(
                'losses', ctc_loss_mean * self.second_task_weight)

            self.summaries_train.append(
                tf.summary.scalar('ctc_loss_train_second',
                                  ctc_loss_mean * self.second_task_weight))
            self.summaries_dev.append(
                tf.summary.scalar('ctc_loss_dev_second',
                                  ctc_loss_mean * self.second_task_weight))

        # Total loss
        self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

        # Add a scalar summary for the snapshot of loss
        with tf.name_scope("total_loss"):
            self.summaries_train.append(
                tf.summary.scalar('total_loss_train', self.loss))
            self.summaries_dev.append(
                tf.summary.scalar('total_loss_dev', self.loss))

        return self.loss

    def decoder(self, decode_type, beam_width=None):
        """Operation for decoding.
        Args:
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
                self.logits_main, tf.cast(self.seq_len, tf.int32))
            decoded_second, _ = tf.nn.ctc_greedy_decoder(
                self.logits_second, tf.cast(self.seq_len, tf.int32))

        elif decode_type == 'beam_search':
            if beam_width is None:
                raise ValueError('Set beam_width.')

            decoded_main, _ = tf.nn.ctc_beam_search_decoder(
                self.logits_main, tf.cast(self.seq_len, tf.int32),
                beam_width=beam_width)
            decoded_second, _ = tf.nn.ctc_beam_search_decoder(
                self.logits_second, tf.cast(self.seq_len, tf.int32),
                beam_width=beam_width)

        decode_op_main = tf.to_int32(decoded_main[0])
        decode_op_second = tf.to_int32(decoded_second[0])

        return decode_op_main, decode_op_second

    def posteriors(self, decode_op_main, decode_op_second):
        """Operation for computing posteriors of each time steps.
        Args:
            decode_op_main: operation for decoding of the main task
            decode_op_second: operation for decoding of the second task
        Return:
            posteriors_op_main: operation for computing posteriors for each
                class in the main task
            posteriors_op_second: operation for computing posteriors for each
                class in the second task
        """
        # logits_3d : (max_time, batch_size, num_classes)
        logits_2d_main = tf.reshape(self.logits_main,
                                    shape=[-1, self.num_classes])
        posteriors_op_main = tf.nn.softmax(logits_2d_main)

        logits_2d_second = tf.reshape(self.logits_second,
                                      shape=[-1, self.num_classes_second])
        posteriors_op_second = tf.nn.softmax(logits_2d_second)

        return posteriors_op_main, posteriors_op_second

    def compute_ler(self, decode_op_main, decode_op_second):
        """Operation for computing LER (Label Error Rate).
        Args:
            decode_op_main: operation for decoding of the main task
            decode_op_second: operation for decoding of the second task
        Return:
            ler_op_main: operation for computing LER of the main task
            ler_op_second: operation for computing LER of the second task
        """
        # Compute LER (normalize by label length)
        ler_op_main = tf.reduce_mean(tf.edit_distance(
            decode_op_main, self.labels, normalize=True))
        ler_op_second = tf.reduce_mean(tf.edit_distance(
            decode_op_second, self.labels_second, normalize=True))
        # TODO: ここでの編集距離はラベルだから，文字に変換しないと正しいCERは得られない

        # Add a scalar summary for the snapshot of LER
        with tf.name_scope("ler"):
            self.summaries_train.append(tf.summary.scalar(
                'ler_train_main', ler_op_main))
            self.summaries_train.append(tf.summary.scalar(
                'ler_train_second', ler_op_second))
            self.summaries_dev.append(tf.summary.scalar(
                'ler_dev_main', ler_op_main))
            self.summaries_dev.append(tf.summary.scalar(
                'ler_dev_second', ler_op_second))

        return ler_op_main, ler_op_second
