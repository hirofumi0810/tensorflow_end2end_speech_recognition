#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class of the multi-task CTC model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from models.ctc.ctc import CTC
from models.encoders.load_encoder import load


class MultitaskCTC(CTC):
    """Hierarchical Connectionist Temporal Classification (CTC) network.
    Args:
        encoder_type (string): The type of an encoder
            multitask_blstm: multitask bidirectional LSTM
            multitask_lstm: multitask unidirectional LSTM
        input_size (int): the dimensions of input vectors
        num_units (int): the number of units in each layer
        num_layers_main (int): the number of layers of the main task
        num_layers_sub (int): the number of layers of the sub task
        num_classes_main (int): the number of classes of target labels in the
            main task (except for a blank label)
        num_classes_second (int): the number of classes of target labels in the
            second task (except for a blank label)
        main_task_weight: A float value. The weight of loss of the main task.
            Set between 0 to 1
        lstm_impl (string, optional): a base implementation of LSTM. This is
            not used for GRU models.
                - BasicLSTMCell: tf.contrib.rnn.BasicLSTMCell (no peephole)
                - LSTMCell: tf.contrib.rnn.LSTMCell
                - LSTMBlockCell: tf.contrib.rnn.LSTMBlockCell
                - LSTMBlockFusedCell: under implementation
                - CudnnLSTM: under implementation
            Choose the background implementation of tensorflow.
            Default is LSTMBlockCell (the fastest).
        use_peephole (bool, optional): if True, use peephole connection. This
            is not used for GRU models.
        splice (int, optional): the number of frames to splice. This is used
            when using CNN-like encoder. Default is 1 frame.
        parameter_init (float, optional): the range of uniform distribution to
            initialize weight parameters (>= 0)
        clip_grad_norm (float, optional): the range of clipping of gradient
            norm (> 0)
        clip_activation (float, optional): the range of clipping of cell
            activation (> 0). This is not used for GRU models.
        num_proj (int, optional): the number of nodes in the projection layer.
            This is not used for GRU models.
        weight_decay (float, optional): a parameter for weight decay
        bottleneck_dim (int, optional): the dimensions of the bottleneck layer
        time_major (bool, optional): if True, time-major computation will be
            performed
    """

    def __init__(self,
                 encoder_type,
                 input_size,
                 num_units,
                 num_layers_main,
                 num_layers_sub,
                 num_classes_main,
                 num_classes_sub,
                 main_task_weight,
                 lstm_impl='LSTMBlockCell',
                 use_peephole=True,
                 splice=1,
                 parameter_init=0.1,
                 clip_grad_norm=None,
                 clip_activation=None,
                 num_proj=None,
                 weight_decay=0.0,
                 bottleneck_dim=None,
                 time_major=True):

        super(MultitaskCTC, self).__init__(
            encoder_type, input_size, num_units, num_layers_main,
            num_classes_main, lstm_impl, use_peephole, splice,
            parameter_init, clip_grad_norm, clip_activation, num_proj,
            weight_decay, bottleneck_dim, time_major)

        self.num_classes_sub = num_classes_sub + 1  # + blank label
        if float(main_task_weight) < 0 or float(main_task_weight) > 1:
            raise ValueError('Set main_task_weight between 0 to 1.')
        self.main_task_weight = main_task_weight
        self.sub_task_weight = 1 - self.main_task_weight

        # Placeholder for multi-task
        self.labels_sub_pl_list = []

        self.name = encoder_type + '_ctc'

        if ['multitask_blstm', 'multitask_lstm']:
            self.encoder = load(encoder_type)(
                num_units=num_units,
                num_proj=self.num_proj,
                num_layers_main=num_layers_main,
                num_layers_sub=num_layers_sub,
                lstm_impl=lstm_impl,
                use_peephole=use_peephole,
                parameter_init=parameter_init,
                clip_activation=clip_activation,
                time_major=time_major)
        else:
            raise NotImplementedError

    def _build(self, inputs, inputs_seq_len, keep_prob):
        """Construct model graph.
        Args:
            inputs: A tensor of size `[B, T, input_size]`
            inputs_seq_len (placeholder): A tensor of size` [B]`
            keep_prob (placeholder, float): A probability to keep nodes
                in the hidden-hidden connection
        Returns:
            logits_main: A tensor of size `[T, B, num_classes]`
            logits_sub: A tensor of size `[T, B, num_classes]`
        """
        # inputs: `[B, T, input_size]`
        batch_size = tf.shape(inputs)[0]
        max_time = tf.shape(inputs)[1]

        encoder_outputs, final_state, encoder_outputs_sub, final_state_sub = self.encoder(
            inputs, inputs_seq_len, keep_prob)

        # Reshape to apply the same weights over the timesteps
        if 'lstm' not in self.encoder_type or self.num_proj is None or self.num_proj == 0:
            if 'b' in self.encoder_type:
                # bidirectional
                outputs_2d = tf.reshape(
                    encoder_outputs, shape=[-1, self.num_units * 2])
                outputs_sub_2d = tf.reshape(
                    encoder_outputs_sub, shape=[-1, self.num_units * 2])
            else:
                # unidirectional
                outputs_2d = tf.reshape(
                    encoder_outputs, shape=[-1, self.num_units])
                outputs_sub_2d = tf.reshape(
                    encoder_outputs_sub, shape=[-1, self.num_units])
        else:
            if 'b' in self.encoder_type:
                # bidirectional
                outputs_2d = tf.reshape(
                    encoder_outputs, shape=[-1, self.num_proj * 2])
                outputs_sub_2d = tf.reshape(
                    encoder_outputs_sub, shape=[-1, self.num_proj * 2])
            else:
                # unidirectional
                outputs_2d = tf.reshape(
                    encoder_outputs, shape=[-1, self.num_proj])
                outputs_sub_2d = tf.reshape(
                    encoder_outputs_sub, shape=[-1, self.num_proj])

        with tf.variable_scope('output_sub') as scope:
            logits_sub_2d = tf.contrib.layers.fully_connected(
                outputs_sub_2d, self.num_classes_sub,
                activation_fn=None,
                weights_initializer=tf.truncated_normal_initializer(
                    stddev=self.parameter_init),
                biases_initializer=tf.zeros_initializer(),
                scope=scope)

            if self.time_major:
                # Reshape back to the original shape
                logits_sub = tf.reshape(
                    logits_sub_2d,
                    shape=[max_time, batch_size, self.num_classes_sub])
            else:
                # Reshape back to the original shape
                logits_sub = tf.reshape(
                    logits_sub_2d,
                    shape=[batch_size, max_time, self.num_classes_sub])

                # Convert to time-major: `[T, B, num_classes]'
                logits = tf.transpose(logits_sub, [1, 0, 2])

        if self.bottleneck_dim is not None and self.bottleneck_dim != 0:
            with tf.variable_scope('bottleneck') as scope:
                outputs_2d = tf.contrib.layers.fully_connected(
                    outputs_2d, self.bottleneck_dim,
                    activation_fn=tf.nn.relu,
                    weights_initializer=tf.truncated_normal_initializer(
                        stddev=self.parameter_init),
                    biases_initializer=tf.zeros_initializer(),
                    scope=scope)

                # Dropout for the hidden-output connections
                outputs_2d = tf.nn.dropout(
                    outputs_2d, keep_prob, name='dropout_bottleneck')

        with tf.variable_scope('output_main') as scope:
            logits_2d = tf.contrib.layers.fully_connected(
                outputs_2d, self.num_classes,
                activation_fn=None,
                weights_initializer=tf.truncated_normal_initializer(
                    stddev=self.parameter_init),
                biases_initializer=tf.zeros_initializer(),
                scope=scope)

            if self.time_major:
                # Reshape back to the original shape
                logits = tf.reshape(
                    logits_2d,
                    shape=[max_time, batch_size, self.num_classes])
            else:
                # Reshape back to the original shape
                logits = tf.reshape(
                    logits_2d,
                    shape=[batch_size, max_time, self.num_classes])

                # Convert to time-major: `[T, B, num_classes]'
                logits = tf.transpose(logits, [1, 0, 2])

        return logits, logits_sub

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
            tf.placeholder(tf.int32, shape=[None], name='inputs_seq_len'))
        self.keep_prob_pl_list.append(
            tf.placeholder(tf.float32, name='keep_prob'))

    def compute_loss(self, inputs, labels_main, labels_sub, inputs_seq_len,
                     keep_prob, scope=None):
        """Operation for computing ctc loss.
        Args:
            inputs: A tensor of size `[B, T, input_size]`
            labels_main: A SparseTensor of target labels in the main task
            labels_sub: A SparseTensor of target labels in the sub task
            inputs_seq_len: A tensor of size `[B]`
            keep_prob (placeholder, float): A probability to keep nodes
                in the hidden-hidden connection
            scope: A scope in the model tower
        Returns:
            total_loss: operation for computing total ctc loss
            logits_main: A tensor of size `[T, B, input_size]`
            logits_sub: A tensor of size `[T, B, input_size]`
        """
        # Build model graph
        logits_main, logits_sub = self._build(
            inputs, inputs_seq_len, keep_prob)

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
                # tf.cast(inputs_seq_len, tf.int32),
                inputs_seq_len,
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
                # tf.cast(inputs_seq_len, tf.int32),
                inputs_seq_len,
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

    def decoder(self, logits_main, logits_sub, inputs_seq_len, beam_width=1):
        """Operation for decoding.
        Args:
            logits_main: A tensor of size `[T, B, input_size]`
            logits_sub: A tensor of size `[T, B, input_size]`
            inputs_seq_len: A tensor of size `[B]`
            beam_width (int, optional): beam width for beam search.
                1 disables beam search, which mean greedy decoding.
        Return:
            decode_op_main: operation for decoding of the main task
            decode_op_sub: operation for decoding of the sub task
        """
        assert isinstance(beam_width, int), "beam_width must be integer."
        assert beam_width >= 1, "beam_width must be >= 1"

        # inputs_seq_len = tf.cast(inputs_seq_len, tf.int32)

        if beam_width == 1:
            decoded_main, _ = tf.nn.ctc_greedy_decoder(
                logits_main, inputs_seq_len)
            decoded_sub, _ = tf.nn.ctc_greedy_decoder(
                logits_sub, inputs_seq_len)

        else:
            decoded_main, _ = tf.nn.ctc_beam_search_decoder(
                logits_main, inputs_seq_len,
                beam_width=beam_width)
            decoded_sub, _ = tf.nn.ctc_beam_search_decoder(
                logits_sub, inputs_seq_len,
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
        # Convert to batch-major: `[B, T, num_classes]'
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
