#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Attention model class based on BLSTM encoder and LSTM decoder.
   This implemtentation is based on
        https://arxiv.org/abs/1609.06773.
          Kim, Suyoun, Takaaki Hori, and Shinji Watanabe.
          "Joint ctc-attention based end-to-end speech recognition using
          multi-task learning." arXiv preprint arXiv:1609.06773 (2016).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from models.attention.attention_seq2seq import AttentionSeq2Seq


class JointCTCAttention(AttentionSeq2Seq):
    """Joint CTC-Attention model. Encoder is BLSTM as in the paper.
    Args:
        input_size (int): the dimension of input vectors
        encoder_type (string): blstm or lstm
        encoder_num_units (int): the number of units in each layer of the
            encoder
        encoder_num_layers (int): the number of layers of the encoder
        encoder_num_proj (int): the number of nodes in the projection layer of
            the encoder. This is not used for GRU encoders.
        attention_type (string): the type of attention
        attention_dim: (int) the dimension of the attention layer
        decoder_type (string): lstm or gru
        decoder_num_units (int): the number of units in each layer of the decoder
        # decoder_num_proj (int): the number of nodes in the projection layer
        # of
            the decoder. This is not used for GRU decoders.
        decoder_num_layers (int): the number of layers of the decoder
        embedding_dim (int): the dimension of the embedding in target spaces
        lambda_weight (float): weight parameter for multi-task training.
            loss = lambda_weight * ctc_loss + \
                (1 - lambda_weight) * attention_loss
        num_classes (int): the number of nodes in softmax layer
        sos_index (int): index of the start of sentence tag (<SOS>)
        eos_index (int): index of the end of sentence tag (<EOS>)
        max_decode_length (int): the length of output sequences to stop
            prediction when EOS token have not been emitted
        lstm_impl (string): a base implementation of LSTM. This is
            not used for GRU models.
                - BasicLSTMCell: tf.contrib.rnn.BasicLSTMCell (no peephole)
                - LSTMCell: tf.contrib.rnn.LSTMCell
                - LSTMBlockCell: tf.contrib.rnn.LSTMBlockCell
                - LSTMBlockFusedCell: under implementation
                - CudnnLSTM: under implementation
            Choose the background implementation of tensorflow.
            Default is LSTMBlockCell.
        use_peephole (bool, optional): if True, use peephole connection. This
            is not used for GRU models.
        splice (int, optional): the number of frames to splice. This is used
            when using CNN-like encoder. Default is 1 frame.
        parameter_init (float, optional): the ange of uniform distribution to
            initialize weight parameters (>= 0)
        clip_grad_norm (float, optional): the range of clipping of gradient
            norm (> 0)
        clip_activation_encoder (float, optional): the range of clipping of
            cell activation of the encoder (> 0). This is not used for GRU
            encoders.
        clip_activation_decoder (float, optional): the range of clipping of
            cell activation of the decoder (> 0). This is not used for GRU
            decoders.
        weight_decay (float, optional): a parameter for weight decay
        time_major (bool, optional): if True, time-major computation will be
            performed
        sharpening_factor (float, optional): a sharpening factor in the
            softmax layer for computing attention weights
        logits_temperature (float, optional): a parameter for smoothing the
            softmax layer in outputing probabilities
    """

    def __init__(self,
                 input_size,
                 encoder_type,
                 encoder_num_units,
                 encoder_num_layers,
                 encoder_num_proj,
                 attention_type,
                 attention_dim,
                 decoder_type,
                 decoder_num_units,
                 #  decoder_num_proj,
                 decoder_num_layers,
                 embedding_dim,
                 lambda_weight,
                 num_classes,
                 sos_index,
                 eos_index,
                 max_decode_length,
                 lstm_impl='LSTMBlockCell',
                 use_peephole=True,
                 splice=1,
                 parameter_init=0.1,
                 clip_grad_norm=5.0,
                 clip_activation_encoder=50,
                 clip_activation_decoder=50,
                 weight_decay=0.0,
                 time_major=True,
                 sharpening_factor=1.0,
                 logits_temperature=1.0,
                 name='joint_ctc_attention'):

        super(JointCTCAttention, self).__init__(
            input_size=input_size,
            encoder_type=encoder_type,
            encoder_num_units=encoder_num_units,
            encoder_num_layers=encoder_num_layers,
            encoder_num_proj=encoder_num_proj,
            attention_type=attention_type,
            attention_dim=attention_dim,
            decoder_type=decoder_type,
            decoder_num_units=decoder_num_units,
            #  decoder_num_proj,
            decoder_num_layers=decoder_num_layers,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            sos_index=sos_index,
            eos_index=eos_index,
            max_decode_length=max_decode_length,
            lstm_impl=lstm_impl,
            use_peephole=use_peephole,
            splice=splice,
            parameter_init=parameter_init,
            clip_grad_norm=clip_grad_norm,
            clip_activation_encoder=clip_activation_encoder,
            clip_activation_decoder=50,
            weight_decay=0.0,
            time_major=True,
            sharpening_factor=1.0,
            logits_temperature=1.0,
            name=name)

        # Setting for multi-task training
        self.ctc_num_classes = num_classes + 1
        self.lambda_weight = lambda_weight

        self.ctc_labels_pl_list = []

    def create_placeholders(self):
        """Create placeholders and append them to list."""
        self.inputs_pl_list.append(
            tf.placeholder(tf.float32, shape=[None, None, self.input_size],
                           name='input'))
        self.labels_pl_list.append(
            tf.placeholder(tf.int32, shape=[None, None], name='labels'))
        self.inputs_seq_len_pl_list.append(
            tf.placeholder(tf.int32, shape=[None], name='inputs_seq_len'))
        self.labels_seq_len_pl_list.append(
            tf.placeholder(tf.int32, shape=[None], name='labels_seq_len'))

        self.keep_prob_encoder_pl_list.append(
            tf.placeholder(tf.float32, name='keep_prob_encoder'))
        self.keep_prob_decoder_pl_list.append(
            tf.placeholder(tf.float32, name='keep_prob_decoder'))
        self.keep_prob_embedding_pl_list.append(
            tf.placeholder(tf.float32, name='keep_prob_embedding'))

        # These are prepared for computing LER
        self.labels_st_true_pl = tf.SparseTensor(
            tf.placeholder(tf.int64, name='indices_true'),
            tf.placeholder(tf.int32, name='values_true'),
            tf.placeholder(tf.int64, name='shape_true'))
        self.labels_st_pred_pl = tf.SparseTensor(
            tf.placeholder(tf.int64, name='indices_pred'),
            tf.placeholder(tf.int32, name='values_pred'),
            tf.placeholder(tf.int64, name='shape_pred'))

        # Placeholder for multi-task training
        self.ctc_labels_pl_list.append(tf.SparseTensor(
            tf.placeholder(tf.int64, name='indices'),
            tf.placeholder(tf.int32, name='values'),
            tf.placeholder(tf.int64, name='shape')))

    def ctc_logits(self, encoder_outputs):
        """
        Args:
            encoder_outputs:
        Returns:
            logits:
        """

        batch_size = tf.shape(encoder_outputs)[0]
        max_time = tf.shape(encoder_outputs)[1]

        # Reshape to apply the same weights over the timesteps
        if 'lstm' not in self.encoder_type or self.encoder_num_proj is None:
            if 'b' in self.encoder_type:
                # bidirectional
                outputs_2d = tf.reshape(
                    encoder_outputs, shape=[-1, self.encoder_num_units * 2])
            else:
                # unidirectional
                outputs_2d = tf.reshape(
                    encoder_outputs, shape=[-1, self.encoder_num_units])
        else:
            if 'b' in self.encoder_type:
                # bidirectional
                outputs_2d = tf.reshape(
                    encoder_outputs, shape=[-1, self.encoder_num_proj * 2])
            else:
                # unidirectional
                outputs_2d = tf.reshape(
                    encoder_outputs, shape=[-1, self.encoder_num_proj])

        with tf.variable_scope('ctc_output') as scope:
            logits_2d = tf.contrib.layers.fully_connected(
                outputs_2d,
                num_outputs=self.ctc_num_classes,
                activation_fn=None,
                weights_initializer=tf.truncated_normal_initializer(
                    stddev=self.parameter_init),
                biases_initializer=tf.zeros_initializer(),
                scope=scope)

            if self.time_major:
                # Reshape back to the original shape
                logits = tf.reshape(
                    logits_2d, shape=[max_time, batch_size, self.ctc_num_classes])
            else:
                # Reshape back to the original shape
                logits = tf.reshape(
                    logits_2d, shape=[batch_size, max_time, self.ctc_num_classes])

                # Convert to time-major: `[T, B, ctc_num_classes]'
                logits = tf.transpose(logits, [1, 0, 2])

        return logits

    def compute_loss(self, inputs, labels, ctc_labels, inputs_seq_len,
                     labels_seq_len,
                     keep_prob_encoder, keep_prob_decoder, keep_prob_embedding,
                     scope=None):
        """Operation for computing cross entropy sequence loss.
        Args:
            inputs: A tensor of `[B, T_in, input_size]`
            labels: A tensor of `[B, T_out]`
            ctc_labels:
            inputs_seq_len: A tensor of `[B]`
            labels_seq_len: A tensor of `[B]`
            keep_prob_encoder (placeholder, float): A probability to keep nodes
                in the hidden-hidden connection of the encoder
            keep_prob_decoder (placeholder, float): A probability to keep nodes
                in the hidden-hidden connection of the decoder
            keep_prob_embedding (placeholder, float): A probability to keep
                nodes in the embedding layer
            scope (optional): A scope in the model tower
        Returns:
            total_loss: operation for computing total loss (cross entropy
                sequence loss + ctc_loss + L2).
                This is a single scalar tensor to minimize.
            logits: A tensor of size `[B, T_in, num_classes + 2 (<SOS> & <EOS>)]`
            ctc_logits: A tensor of size `[B, T_in, num_classes + 1 (blank)]`
            decoder_outputs_train (namedtuple): A namedtuple of
                `(logits, predicted_ids, decoder_output, attention_weights,
                    context_vector)`
            decoder_outputs_infer (namedtuple): A namedtuple of
                `(logits, predicted_ids, decoder_output, attention_weights,
                    context_vector)`
        """
        # Build model graph
        logits, decoder_outputs_train, decoder_outputs_infer, encoder_outputs = self._build(
            inputs, labels, inputs_seq_len, labels_seq_len,
            keep_prob_encoder, keep_prob_decoder, keep_prob_embedding)

        # For prevent 0 * log(0) in crossentropy loss
        epsilon = tf.constant(value=1e-10)
        logits = logits + epsilon

        # Weight decay
        if self.weight_decay > 0:
            with tf.name_scope("weight_decay_loss"):
                weight_sum = 0
                for var in tf.trainable_variables():
                    if 'bias' not in var.name.lower():
                        weight_sum += tf.nn.l2_loss(var)
                tf.add_to_collection('losses', weight_sum * self.weight_decay)

        with tf.name_scope("sequence_loss"):
            # batch_size = tf.cast(tf.shape(inputs)[0], tf.float32)
            labels_max_seq_len = tf.shape(labels[:, 1:])[1]
            loss_mask = tf.sequence_mask(tf.to_int32(labels_seq_len - 1),
                                         maxlen=labels_max_seq_len,
                                         dtype=tf.float32)
            sequence_loss = tf.contrib.seq2seq.sequence_loss(
                logits=logits,
                targets=labels[:, 1:],  # exclude <SOS>
                weights=loss_mask,
                average_across_timesteps=True,
                average_across_batch=True,
                softmax_loss_function=None)
            # sequence_loss /= batch_size

            tf.add_to_collection('losses', sequence_loss *
                                 (1 - self.lambda_weight))

        with tf.name_scope('ctc_loss'):

            ctc_logits = self.ctc_logits(encoder_outputs)

            ctc_losses = tf.nn.ctc_loss(
                ctc_labels,  # NOTE: sparsetensor
                ctc_logits,
                # tf.cast(inputs_seq_len, tf.int32),
                inputs_seq_len,
                preprocess_collapse_repeated=False,
                ctc_merge_repeated=True,
                ignore_longer_outputs_than_inputs=False,
                time_major=True)
            ctc_loss = tf.reduce_mean(ctc_losses, name='ctc_loss_mean')
            tf.add_to_collection('losses', ctc_loss * self.lambda_weight)

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
            tf.summary.scalar('sequence_loss_train', sequence_loss))
        self.summaries_dev.append(
            tf.summary.scalar('sequence_loss_dev', sequence_loss))
        self.summaries_train.append(
            tf.summary.scalar('ctc_loss_train', ctc_loss))
        self.summaries_dev.append(
            tf.summary.scalar('ctc_loss_dev', ctc_loss))

        return total_loss, logits, ctc_logits, decoder_outputs_train, decoder_outputs_infer
