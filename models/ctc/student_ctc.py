#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Student CTC model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from models.model_base import ModelBase

from models.encoders.core.student_cnn_ctc import StudentCNNCTCEncoder
from models.encoders.core.student_cnn_compact_ctc import StudentCNNCompactCTCEncoder
from models.encoders.core.student_cnn_xe import StudentCNNXEEncoder
from models.encoders.core.student_cnn_compact_xe import StudentCNNCompactXEEncoder


class StudentCTC(ModelBase):
    """Connectionist Temporal Classification (CTC) network.
    Args:
        encoder_type (string): The type of an encoder
            student_cnn_xe:
            student_cnn_compact_xe:
            student_cnn_ctc:
            student_cnn_compact_ctc:
        input_size (int): the dimensions of input vectors
        num_classes (int): the number of classes of target labels
            (except for a blank label)
        splice (int, optional): the number of frames to splice. This is used
            when using CNN-like encoder. Default is 1 frame.
        num_stack (int, optional): the number of frames to stack
        parameter_init (float, optional): the range of uniform distribution to
            initialize weight parameters (>= 0)
        clip_grad_norm (float, optional): the range of clipping of gradient
            norm (> 0)
        weight_decay (float, optional): a parameter for weight decay
        time_major (bool, optional): if True, time-major computation will be
            performed
    """

    def __init__(self,
                 encoder_type,
                 input_size,
                 num_classes,
                 splice=1,
                 num_stack=1,
                 parameter_init=0.1,
                 clip_grad_norm=None,
                 weight_decay=0.0,
                 time_major=True):

        super(StudentCTC, self).__init__()

        assert input_size % 3 == 0, 'input_size must be divisible by 3 (+ delta, double delta features).'
        assert splice % 2 == 1, 'splice must be the odd number'
        if clip_grad_norm is not None:
            assert float(
                clip_grad_norm) > 0, 'clip_grad_norm must be larger than 0.'
        assert float(
            weight_decay) >= 0, 'weight_decay must not be a negative value.'

        # Encoder setting
        self.encoder_type = encoder_type
        self.input_size = input_size
        self.splice = splice
        self.num_stack = num_stack  # NOTE: this is used for CNN-like encoders
        self.num_classes = num_classes + 1  # + blank

        # Regularization
        self.parameter_init = parameter_init
        self.clip_grad_norm = clip_grad_norm
        self.weight_decay = weight_decay

        # Summaries for TensorBoard
        self.summaries_train = []
        self.summaries_dev = []

        # Placeholders
        self.inputs_pl_list = []
        self.labels_pl_list = []
        self.inputs_seq_len_pl_list = []
        self.keep_prob_pl_list = []

        self.time_major = time_major
        self.name = encoder_type + '_ctc'

        if encoder_type == 'student_cnn':
            self.encoder = StudentCNNCTCEncoder(
                input_size=input_size,
                splice=splice,
                num_stack=num_stack,
                parameter_init=parameter_init,
                time_major=time_major)

        elif encoder_type == 'student_cnn_compact':
            self.encoder = StudentCNNCompactCTCEncoder(
                input_size=input_size,
                splice=splice,
                num_stack=num_stack,
                parameter_init=parameter_init,
                time_major=time_major)

        elif encoder_type == 'student_cnn_xe':
            self.encoder = StudentCNNXEEncoder(
                input_size=input_size,
                splice=splice,
                num_stack=num_stack,
                parameter_init=parameter_init)

        elif encoder_type == 'student_cnn_compact_xe':
            self.encoder = StudentCNNCompactXEEncoder(
                input_size=input_size,
                splice=splice,
                num_stack=num_stack,
                parameter_init=parameter_init)

        else:
            raise NotImplementedError

    def _build_ctc(self, inputs, inputs_seq_len, keep_prob, is_training):
        """Construct model graph.
        Args:
            inputs: A tensor of size `[B, T, input_size]`
            inputs_seq_len (placeholder): A tensor of size` [B]`
            keep_prob (placeholder, float): A probability to keep nodes
                in the hidden-hidden connection
            is_training (bool):
        Returns:
            logits: A tensor of size `[T, B, num_classes]`
        """
        # inputs: `[B, T, input_size]`
        batch_size = tf.shape(inputs)[0]
        max_time = tf.shape(inputs)[1]

        encoder_outputs, final_state = self.encoder(
            inputs, inputs_seq_len, keep_prob, is_training)

        # Reshape to apply the same weights over the timesteps
        output_dim = encoder_outputs.shape.as_list()[-1]
        outputs_2d = tf.reshape(
            encoder_outputs, shape=[batch_size * max_time, output_dim])

        with tf.variable_scope('output') as scope:
            logits_2d = tf.contrib.layers.fully_connected(
                outputs_2d,
                num_outputs=self.num_classes,
                activation_fn=None,
                weights_initializer=tf.truncated_normal_initializer(
                    stddev=self.parameter_init),
                biases_initializer=tf.zeros_initializer(),
                scope=scope)

            if self.time_major:
                # Reshape back to the original shape
                logits = tf.reshape(
                    logits_2d, shape=[max_time, batch_size, self.num_classes])
            else:
                # Reshape back to the original shape
                logits = tf.reshape(
                    logits_2d, shape=[batch_size, max_time, self.num_classes])

                # Convert to time-major: `[T, B, num_classes]'
                logits = tf.transpose(logits, [1, 0, 2])

        return logits

    def _build_xe(self, inputs, keep_prob, is_training):
        """Construct model graph.
        Args:
            inputs: A tensor of size `[B, input_size]`
            keep_prob (placeholder, float): A probability to keep nodes
                in the hidden-hidden connection
            is_training (bool):
        Returns:
            logits: A tensor of size `[B, num_classes]`
        """
        encoder_outputs = self.encoder(inputs, keep_prob, is_training)

        with tf.variable_scope('output') as scope:
            logits = tf.contrib.layers.fully_connected(
                encoder_outputs,
                num_outputs=self.num_classes,
                activation_fn=None,
                weights_initializer=tf.truncated_normal_initializer(
                    stddev=self.parameter_init),
                biases_initializer=tf.zeros_initializer(),
                scope=scope)

        return logits

    def create_placeholders_ctc(self):
        """Create placeholders for CTC training and append them to list."""
        self.inputs_pl_list.append(
            tf.placeholder(tf.float32,
                           shape=[None, None, self.input_size * self.splice],
                           name='input'))
        self.labels_pl_list.append(
            tf.SparseTensor(tf.placeholder(tf.int64, name='indices'),
                            tf.placeholder(tf.int32, name='values'),
                            tf.placeholder(tf.int64, name='shape')))
        self.inputs_seq_len_pl_list.append(
            tf.placeholder(tf.int32, shape=[None], name='inputs_seq_len'))
        self.keep_prob_pl_list.append(
            tf.placeholder(tf.float32, name='keep_prob'))

    def create_placeholders_xe(self):
        """Create placeholders for XE training and append them to list."""
        self.inputs_pl_list.append(
            tf.placeholder(tf.float32,
                           shape=[None, self.input_size],
                           name='input'))
        self.labels_pl_list.append(
            tf.placeholder(tf.float32,
                           shape=[None, self.num_classes],
                           name='label'))
        self.keep_prob_pl_list.append(
            tf.placeholder(tf.float32, name='keep_prob'))

    def compute_ctc_loss(self, inputs, labels, inputs_seq_len,
                         keep_prob, scope=None, softmax_temperature=1,
                         is_training=True):
        """Operation for computing CTC loss.
        Args:
            inputs: A tensor of size `[B, T, input_size]`
            labels: A SparseTensor of target labels
            inputs_seq_len: A tensor of size `[B]`
            keep_prob (placeholder, float): A probability to keep nodes
                in the hidden-hidden connection
            scope (optional): A scope in the model tower
            softmax_temperature (int, optional): temperature parameter for
                ths softmax layer in the student training stage
            is_training (bool, optional):
        Returns:
            total_loss: operation for computing total ctc loss (ctc loss + L2).
                 This is a single scalar tensor to minimize.
            logits: A tensor of size `[T, B, num_classes]`
        """
        # Build model graph
        logits = self._build_ctc(inputs, inputs_seq_len, keep_prob,
                                 is_training=is_training)

        # Weight decay
        if self.weight_decay > 0:
            with tf.name_scope("weight_decay_loss"):
                weight_sum = 0
                for var in tf.trainable_variables():
                    if 'bias' not in var.name.lower():
                        weight_sum += tf.nn.l2_loss(var)
                tf.add_to_collection('losses', weight_sum * self.weight_decay)

        with tf.name_scope("ctc_loss"):
            ctc_losses = tf.nn.ctc_loss(
                labels,
                logits / softmax_temperature,
                tf.cast(inputs_seq_len, tf.int32),
                # inputs_seq_len,
                preprocess_collapse_repeated=False,
                ctc_merge_repeated=True,
                # ignore_longer_outputs_than_inputs=False,
                ignore_longer_outputs_than_inputs=True,
                time_major=True)
            ctc_loss = tf.reduce_mean(ctc_losses, name='ctc_loss_mean')
            tf.add_to_collection('losses', ctc_loss)

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
            tf.summary.scalar('ctc_loss_train', ctc_loss))
        self.summaries_dev.append(
            tf.summary.scalar('ctc_loss_dev', ctc_loss))

        return total_loss, logits

    def compute_xe_loss(self, inputs, soft_targets, keep_prob,
                        scope=None, softmax_temperature=1,
                        is_training=True):
        """Operation for computing XE loss.
        Args:
            inputs: A tensor of size `[B, input_size]`
            soft_targets: A tensor of size `[B, num_classes]`
            keep_prob (placeholder, float): A probability to keep nodes
                in the hidden-hidden connection
            scope (optional): A scope in the model tower
            softmax_temperature (int, optional): temperature parameter for
                ths softmax layer in the student training stage
            is_training (bool, optional):
        Returns:
            total_loss: operation for computing total ctc loss (XE loss + L2).
                 This is a single scalar tensor to minimize.
            logits: A tensor of size `[B, num_classes]`
        """
        # Build model graph
        logits = self._build_xe(inputs, keep_prob,
                                is_training=is_training)

        # Weight decay
        if self.weight_decay > 0:
            with tf.name_scope("weight_decay_loss"):
                weight_sum = 0
                for var in tf.trainable_variables():
                    if 'bias' not in var.name.lower():
                        weight_sum += tf.nn.l2_loss(var)
                tf.add_to_collection('losses', weight_sum * self.weight_decay)

        with tf.name_scope("xe_loss"):
            losses_soft = tf.nn.softmax_cross_entropy_with_logits(
                labels=soft_targets,
                # logits=logits / (softmax_temperature ** 2),
                logits=logits)
            loss_soft = tf.reduce_mean(losses_soft, name='xe_loss_mean')
            tf.add_to_collection('losses', loss_soft)

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
            tf.summary.scalar('xe_loss_train', loss_soft))
        self.summaries_dev.append(
            tf.summary.scalar('xe_loss_dev', loss_soft))

        return total_loss, logits

    def decoder(self, logits, inputs_seq_len, beam_width=1):
        """Operation for decoding.
        Args:
            logits: A tensor of size `[T, B, num_classes]`
            inputs_seq_len: A tensor of size `[B]`
            beam_width (int, optional): beam width for beam search.
                1 disables beam search, which mean greedy decoding.
        Return:
            decode_op: A SparseTensor
        """
        assert isinstance(beam_width, int), "beam_width must be integer."
        assert beam_width >= 1, "beam_width must be >= 1"

        # inputs_seq_len = tf.cast(inputs_seq_len, tf.int32)

        if beam_width == 1:
            decoded, _ = tf.nn.ctc_greedy_decoder(
                logits, inputs_seq_len)
        else:
            decoded, _ = tf.nn.ctc_beam_search_decoder(
                logits, inputs_seq_len,
                beam_width=beam_width)

        decode_op = tf.to_int32(decoded[0])

        # TODO: chnage function name to `decode`

        return decode_op

    def posteriors(self, logits, blank_prior=1):
        """Operation for computing posteriors of each time steps.
        Args:
            logits: A tensor of size `[T, B, num_classes]`
            blank_prior (float): A prior for blank classes. posteriors are
                divided by this prior.
        Return:
            posteriors_op: operation for computing posteriors for each class
        """
        # Convert to batch-major: `[B, T, num_classes]'
        logits = tf.transpose(logits, (1, 0, 2))

        logits_2d = tf.reshape(logits, [-1, self.num_classes])

        posteriors_op = tf.nn.softmax(logits_2d)

        return posteriors_op

    def compute_ler(self, decode_op, labels):
        """Operation for computing LER (Label Error Rate).
        Args:
            decode_op: operation for decoding
            labels: A SparseTensor of target labels
        Return:
            ler_op: operation for computing LER
        """
        # Compute LER (normalize by label length)
        ler_op = tf.reduce_mean(tf.edit_distance(
            decode_op, labels, normalize=True))

        # Add a scalar summary for the snapshot of LER
        self.summaries_train.append(tf.summary.scalar('ler_train', ler_op))
        self.summaries_dev.append(tf.summary.scalar('ler_dev', ler_op))

        return ler_op
