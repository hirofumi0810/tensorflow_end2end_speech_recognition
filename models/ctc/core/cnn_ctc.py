#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""CNN-CTC model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from models.ctc.core.ctc_base import ctcBase

# from ..feed_forward.layers.activation_func import maxout, prelu
# from ..feed_forward.utils_cnn import pool


class CNN_CTC(ctcBase):
    """CNN-CTC model.
       This implementaion is based on
           https://arxiv.org/abs/1701.02720.
               Zhang, Ying, et al.
               "Towards end-to-end speech recognition with deep convolutional
                neural networks."
               arXiv preprint arXiv:1701.02720 (2017).
    Args:
        input_size: int, the dimensions of input vectors
        num_unit: int, the number of units in each layer
        num_layer: int, the number of layers
        num_classes: int, the number of classes of target labels
            (except for a blank label)
        parameter_init: A float value. Range of uniform distribution to
            initialize weight parameters
        clip_grad: A float value. Range of gradient clipping (> 0)
        clip_activation: A float value. Range of activation clipping (> 0)
        dropout_ratio_input: A float value. Dropout ratio in the input-hidden
            layer
        dropout_ratio_hidden: A float value. Dropout ratio in the hidden-hidden
            layers
        dropout_ratio_output: A float value. Dropout ratio in the hidden-output
            layer
        num_proj: not used
        weight_decay: A float value. Regularization parameter for weight decay
        bottleneck_dim: not used
    """

    def __init__(self,
                 input_size,
                 num_unit,  # TODO: not used
                 num_layer,
                 num_classes,
                 parameter_init=0.1,
                 clip_grad=None,
                 clip_activation=None,
                 dropout_ratio_input=1.0,
                 dropout_ratio_hidden=1.0,
                 dropout_ratio_output=1.0,
                 num_proj=None,  # not used
                 weight_decay=0.0,
                 bottleneck_dim=None,  # not used
                 name='cnn_ctc'):

        ctcBase.__init__(self, input_size, num_unit, num_layer, num_classes,
                         parameter_init, clip_grad, clip_activation,
                         dropout_ratio_input, dropout_ratio_hidden,
                         dropout_ratio_output, weight_decay, name)

        self.num_proj = None
        self.splice = 0

    def _build(self, inputs, inputs_seq_len, keep_prob_input,
               keep_prob_hidden, keep_prob_output):
        """Construct model graph.
        Args:
            inputs: A tensor of size `[batch_size, max_time, input_dim]`
            inputs_seq_len:  A tensor of size `[batch_size]`
            keep_prob_input: A float value. A probability to keep nodes in
                the input-hidden layer
            keep_prob_hidden: A float value. A probability to keep nodes in
                the hidden-hidden layers
            keep_prob_output: A float value. A probability to keep nodes in
                the hidden-output layer
        Returns:
            logits: A tensor of size `[max_time, batch_size, num_classes]`
        """
        # Dropout for inputs
        keep_prob_input = tf.placeholder(tf.float32,
                                         name='keep_prob_input')
        keep_prob_hidden = tf.placeholder(tf.float32,
                                          name='keep_prob_hidden')
        outputs = tf.nn.dropout(inputs,
                                keep_prob_input,
                                name='dropout_input')

        # `[batch_size, max_time, input_size_splice]`
        inputs_shape = tf.shape(inputs)
        batch_size, max_time = inputs_shape[0], inputs_shape[1]

        ######################################################
        # 1st conv
        # filter: (freq,time)=(3,5)
        # W: [FH, FW, InputChannel, FilterNum (OutputChannel)]
        # b: [FilterNum (OutputChannel)]
        ######################################################
        with tf.name_scope('conv1'):
            # Reshape to [batch, 40fbank + 1energy, timesteps, 3(current +
            # delta + deltadelta)]
            input_drop_rs = tf.reshape(
                self.inputs,
                # shape=[batch_size, self.input_size,
                # max_time, 1])
                shape=[batch_size, int(self.input_size / 3), max_time, 3])

            # Affine
            conv1_shape = [3, 5, 3, 128]  # (FH, FW, InputChannel, FilterNum)
            W_conv1 = tf.Variable(tf.truncated_normal(
                shape=conv1_shape, stddev=self.parameter_init))
            b_conv1 = tf.Variable(tf.zeros(
                shape=[conv1_shape[3]]))
            outputs = tf.nn.bias_add(tf.nn.conv2d(input_drop_rs, W_conv1,
                                                  strides=[1, 1, 1, 1],
                                                  padding='SAME'),
                                     b_conv1)

            # Weight decay
            # self._weight_decay(W_conv1)

            # Batch normalization
            # outputs = self._batch_norm(outputs)

            # Activation
            outputs = tf.nn.relu(outputs)

        #########################
        # 1st pool (3*1)
        #########################
        with tf.name_scope('pool1'):
            outputs = pool(outputs, shape=[3, 1], pool_type='max')

            # Dropout
            outputs = tf.nn.dropout(outputs, keep_prob_hidden)

        ############################
        # 2~4th conv
        # filter: (freq,time)=(3,5)
        ############################
        for i_layer in range(2, 5, 1):
            with tf.name_scope('conv' + str(i_layer)):
                # Affine
                if i_layer != 4:
                    # (FH, FW, InputChannel, FilterNum)
                    conv2_shape = [3, 5, 128, 128]
                else:
                    # (FH, FW, InputChannel, FilterNum)
                    conv2_shape = [3, 5, 128, 256]
                W_conv2 = tf.Variable(tf.truncated_normal(
                    shape=conv2_shape, stddev=self.parameter_init))
                b_conv2 = tf.Variable(tf.zeros(
                    shape=[conv2_shape[3]]))
                outputs = tf.nn.bias_add(tf.nn.conv2d(outputs, W_conv2,
                                                      strides=[1, 1, 1, 1],
                                                      padding='SAME'),
                                         b_conv2)

                # Weight decay
                # self._weight_decay(W_conv2)

                # Batch normalization
                # output_conv2 = self._batch_norm(outputs)

                # Activation
                outputs = tf.nn.relu(outputs)

                # Dropout
                outputs = tf.nn.dropout(outputs, keep_prob_hidden)

        ###########################
        # 5~10th conv
        # filter: (freq,time)=(3,5)
        ###########################
        for i_layer in range(5, 11, 1):
            with tf.name_scope('conv' + str(i_layer)):
                # Affine
                # (FH, FW, InputChannel, FilterNum)
                conv5_shape = [3, 5, 256, 256]
                W_conv5 = tf.Variable(tf.truncated_normal(
                    shape=conv5_shape, stddev=self.parameter_init))
                b_conv5 = tf.Variable(tf.zeros(
                    shape=[conv5_shape[3]]))
                outputs = tf.nn.bias_add(tf.nn.conv2d(outputs, W_conv5,
                                                      strides=[1, 1, 1, 1],
                                                      padding='SAME'),
                                         b_conv5)

                # Weight decay
                # self._weight_decay(W_conv5)

                # Batch normalization
                # outputs_conv5 = self._batch_norm(outputs)

                # Activation
                outputs = tf.nn.relu(outputs)

                # Dropout
                outputs = tf.nn.dropout(outputs, keep_prob_hidden)

        # Reshape for fully-connected layer
        outputs = tf.reshape(outputs, shape=[-1, 14 * 256])

        ##############
        # 11~13th fc
        ##############
        for i_layer in range(11, 14, 1):
            with tf.name_scope('fc' + str(i_layer)):
                if i_layer == 11:
                    fc_shape = [14 * 256, 1024]
                elif i_layer == 12:
                    fc_shape = [1024, 1024]
                elif i_layer == 13:
                    fc_shape = [1024, self.num_classes]
                W_fc11 = tf.Variable(tf.truncated_normal(
                    shape=fc_shape, stddev=self.parameter_init))
                b_fc11 = tf.Variable(tf.zeros(
                    shape=[fc_shape[1]]))
                outputs = tf.matmul(outputs, W_fc11) + b_fc11

                # Weight decay
                # self._weight_decay(W_fc11)

                # Batch normalization
                # output = self._batch_norm(outputs)

                # Activation
                outputs = tf.nn.relu(outputs)

                # Dropout
                if i_layer != 13:
                    outputs = tf.nn.dropout(outputs, keep_prob_hidden)

        # Reshape back to the original shape
        logits = tf.reshape(
            logits_2d, shape=[batch_size, -1, self.num_classes])

        # Convert to time-major: `[max_time, batch_size, num_classes]'
        logits = tf.transpose(logits, (1, 0, 2))

        # Dropout for the hidden-output connections
        logits = tf.nn.dropout(logits,
                               keep_prob_output,
                               name='dropout_output')

        return logits
