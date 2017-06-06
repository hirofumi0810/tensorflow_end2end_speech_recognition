#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Multi-task Bidirectional LSTM-CTC model."""

import tensorflow as tf
from .ctc_base import ctcBase


class Multitask_BLSTM_CTC(ctcBase):
    """Multi-task Bidirectional LSTM-CTC model.
    Args:
        batch_size: int, batch size of mini batch
        input_size: int, the dimensions of input vectors
        num_cell: int, the number of memory cells in each layer
        num_layer: int, the number of layers
        output_size: int, the number of nodes in softmax layer of the main task (except for blank class)
        output_size2: int, the number of nodes in softmax layer of the second task (except for blank class)
        main_task_weight: A float value. The weight of loss of the main task. Set between 0 to 1
        parameter_init: A float value. Range of uniform distribution to initialize weight parameters
        clip_gradients: A float value. Range of gradient clipping (non-negative)
        clip_activation: A float value. Range of activation clipping (non-negative)
        dropout_ratio_input: A float value. Dropout ratio in input-hidden layers
        dropout_ratio_hidden: A float value. Dropout ratio in hidden-hidden layers
        num_proj: int, the number of nodes in recurrent projection layer
        weight_decay: A float value. Regularization parameter for weight decay
        bottleneck_dim: not used
        num_layer2: not used
    """

    def __init__(self,
                 batch_size,
                 input_size,
                 num_cell,
                 num_layer,
                 output_size,
                 output_size2,
                 main_task_weight,
                 parameter_init=0.1,
                 clip_gradients=None,
                 clip_activation=None,
                 dropout_ratio_input=1.0,
                 dropout_ratio_hidden=1.0,
                 num_proj=None,
                 weight_decay=0.0,
                 bottleneck_dim=None,
                 num_layer2=None):

        ctcBase.__init__(self, batch_size, input_size, num_cell, num_layer,
                         output_size, parameter_init,
                         clip_gradients, clip_activation,
                         dropout_ratio_input, dropout_ratio_hidden,
                         weight_decay)

        self.num_proj = None if num_proj == 0 else num_proj

        self.num_classes2 = output_size2 + 1  # plus blank label

        if main_task_weight < 0 or main_task_weight > 1:
            raise ValueError('Set main_task_weight between 0 to 1.')
        self.main_task_weight = main_task_weight
        self.second_task_weight = 1 - main_task_weight

    def define(self):
        """Construct Bidirectional LSTM layers."""
        # Generate placeholders
        self._generate_pl()
        self.label_indices_pl2 = tf.placeholder(tf.int64, name='indices')
        self.label_values_pl2 = tf.placeholder(tf.int32, name='values')
        self.label_shape_pl2 = tf.placeholder(tf.int64, name='shape')
        self.labels_pl2 = tf.SparseTensor(self.label_indices_pl2,
                                          self.label_values_pl2,
                                          self.label_shape_pl2)

        # Dropout for Input
        self.inputs = tf.nn.dropout(self.inputs_pl,
                                    self.keep_prob_input_pl,
                                    name='dropout_input')

        # Hidden layers
        outputs = self.inputs
        for i_layer in range(self.num_layer):
            with tf.name_scope('BiLSTM_hidden' + str(i_layer + 1)):

                initializer = tf.random_uniform_initializer(minval=-self.parameter_init,
                                                            maxval=self.parameter_init)

                lstm_fw = tf.contrib.rnn.LSTMCell(self.num_cell,
                                                  use_peepholes=True,
                                                  cell_clip=self.clip_activation,
                                                  initializer=initializer,
                                                  num_proj=self.num_proj,
                                                  forget_bias=1.0,
                                                  state_is_tuple=True)
                lstm_bw = tf.contrib.rnn.LSTMCell(self.num_cell,
                                                  use_peepholes=True,
                                                  cell_clip=self.clip_activation,
                                                  initializer=initializer,
                                                  num_proj=self.num_proj,
                                                  forget_bias=1.0,
                                                  state_is_tuple=True)

                # Dropout (output)
                lstm_fw = tf.contrib.rnn.DropoutWrapper(lstm_fw,
                                                        output_keep_prob=self.keep_prob_hidden_pl)
                lstm_bw = tf.contrib.rnn.DropoutWrapper(lstm_bw,
                                                        output_keep_prob=self.keep_prob_hidden_pl)

                # _init_state_fw = lstm_fw.zero_state(self.batch_size, tf.float32)
                # _init_state_bw = lstm_bw.zero_state(self.batch_size, tf.float32)
                # initial_state_fw=_init_state_fw,
                # initial_state_bw=_init_state_bw,

                # Ignore 2nd return (the last state)
                (outputs_fw, outputs_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=lstm_fw,
                    cell_bw=lstm_bw,
                    inputs=outputs,
                    sequence_length=self.seq_len_pl,
                    dtype=tf.float32,
                    scope='BiLSTM_' + str(i_layer + 1))

                outputs = tf.concat(axis=2, values=[outputs_fw, outputs_bw])

        # Reshape to apply the same weights over the timesteps
        if self.num_proj is None:
            output_node = self.num_cell * 2
        else:
            output_node = self.num_proj * 2
        outputs = tf.reshape(outputs, shape=[-1, output_node])

        # (batch_size, max_timesteps, input_size_splice)
        inputs_shape = tf.shape(self.inputs_pl)
        batch_size, max_timesteps = inputs_shape[0], inputs_shape[1]

        with tf.name_scope('output1'):
            # Affine
            W_output1 = tf.Variable(tf.truncated_normal(shape=[output_node, self.num_classes],
                                                        stddev=0.1, name='W_output1'))
            b_output1 = tf.Variable(
                tf.zeros(shape=[self.num_classes], name='b_output1'))
            logits1_2d = tf.matmul(outputs, W_output1) + b_output1

            # Reshape back to the original shape
            logits1_3d = tf.reshape(
                logits1_2d, shape=[batch_size, -1, self.num_classes])

            # Convert to (max_timesteps, batch_size, num_classes)
            self.logits1 = tf.transpose(logits1_3d, (1, 0, 2))

        with tf.name_scope('output2'):
            # Affine
            W_output2 = tf.Variable(tf.truncated_normal(shape=[output_node, self.num_classes2],
                                                        stddev=0.1, name='W_output2'))
            b_output2 = tf.Variable(
                tf.zeros(shape=[self.num_classes2], name='b_output2'))
            logits2_2d = tf.matmul(outputs, W_output2) + b_output2

            # Reshape back to the original shape
            logits2_3d = tf.reshape(
                logits2_2d, shape=[batch_size, -1, self.num_classes2])

            # Convert to (max_timesteps, batch_size, num_classes)
            self.logits2 = tf.transpose(logits2_3d, (1, 0, 2))

    def loss(self):
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

        with tf.name_scope("ctc_loss1"):
            ctc_loss1 = tf.nn.ctc_loss(
                self.labels_pl, self.logits1, tf.cast(self.seq_len_pl, tf.int32))
            ctc_loss1_mean = tf.reduce_mean(ctc_loss1, name='ctc_loss1_mean')
            tf.add_to_collection(
                'losses', ctc_loss1_mean * self.main_task_weight)

            self.summaries_train.append(
                tf.summary.scalar('loss_train', ctc_loss1_mean * self.main_task_weight))
            self.summaries_dev.append(
                tf.summary.scalar('loss_dev', ctc_loss1_mean * self.main_task_weight))

        with tf.name_scope("ctc_loss2"):
            ctc_loss2 = tf.nn.ctc_loss(
                self.labels_pl2, self.logits2, tf.cast(self.seq_len_pl, tf.int32))
            ctc_loss2_mean = tf.reduce_mean(ctc_loss2, name='ctc_loss2_mean2')
            tf.add_to_collection(
                'losses', ctc_loss2_mean * self.second_task_weight)

            self.summaries_train.append(
                tf.summary.scalar('loss_train', ctc_loss2_mean * self.second_task_weight))
            self.summaries_dev.append(
                tf.summary.scalar('loss_dev', ctc_loss2_mean * self.second_task_weight))

        # Total loss
        self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

        # Add a scalar summary for the snapshot of loss
        with tf.name_scope("total_loss"):
            self.summaries_train.append(
                tf.summary.scalar('loss_train', self.loss))
            self.summaries_dev.append(
                tf.summary.scalar('loss_dev', self.loss))

        return self.loss

    def decoder(self, decode_type, beam_width=None):
        """Operation for decoding.
        Args:
            decode_type: greedy or beam_search
            beam_width: beam width for beam search
        Return:
            decode_op1: operation for decoding of the main task
            decode_op22: operation for decoding of the second task
        """
        if decode_type not in ['greedy', 'beam_search']:
            raise ValueError('decode_type is "greedy" or "beam_search".')

        if decode_type == 'greedy':
            decoded1, _ = tf.nn.ctc_greedy_decoder(
                self.logits1, tf.cast(self.seq_len_pl, tf.int32))
            decoded2, _ = tf.nn.ctc_greedy_decoder(
                self.logits2, tf.cast(self.seq_len_pl, tf.int32))

        elif decode_type == 'beam_search':
            if beam_width is None:
                raise ValueError('Set beam_width.')

            decoded1, _ = tf.nn.ctc_beam_search_decoder(self.logits1,
                                                        tf.cast(
                                                            self.seq_len_pl, tf.int32),
                                                        beam_width=beam_width)
            decoded2, _ = tf.nn.ctc_beam_search_decoder(self.logits2,
                                                        tf.cast(
                                                            self.seq_len_pl, tf.int32),
                                                        beam_width=beam_width)

        decode_op1 = tf.to_int32(decoded1[0])
        decode_op2 = tf.to_int32(decoded2[0])

        return decode_op1, decode_op2

    def ler(self, decode_op1, decode_op2):
        """Operation for computing LER.
        Args:
            decode_op1: operation for decoding of the main task
            decode_op2: operation for decoding of the second task
        Return:
            ler_op1: operation for computing label error rate of the main task
            ler_op2: operation for computing label error rate of the second task
        """
        # Compute label error rate (normalize by label length)
        ler_op1 = tf.reduce_mean(tf.edit_distance(
            decode_op1, self.labels_pl, normalize=True))
        ler_op2 = tf.reduce_mean(tf.edit_distance(
            decode_op2, self.labels_pl2, normalize=True))
        # TODO: ここでの編集距離はラベルだから，文字に変換しないと正しいCERは得られない

        # Add a scalar summary for the snapshot of ler
        with tf.name_scope("ler"):
            self.summaries_train.append(tf.summary.scalar(
                'ler_train_main', ler_op1))
            self.summaries_train.append(tf.summary.scalar(
                'ler_train_second', ler_op2))
            self.summaries_dev.append(tf.summary.scalar(
                'ler_dev_main', ler_op1))
            self.summaries_dev.append(tf.summary.scalar(
                'ler_dev_second', ler_op2))

        return ler_op1, ler_op2
