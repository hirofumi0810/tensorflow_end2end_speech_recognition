#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""CNN-CTC model."""

import tensorflow as tf
from .ctc_base import ctcBase

# from ..feed_forward.layers.activation_func import maxout, prelu
# from ..feed_forward.utils_cnn import pool


class CNN_CTC(ctcBase):
    """CNN-CTC model.
       This implementaion is based on
           https://arxiv.org/abs/1701.02720.
               Zhang, Ying, et al.
               "Towards end-to-end speech recognition with deep convolutional neural networks."
               arXiv preprint arXiv:1701.02720 (2017).
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
        num_proj: int, the number of nodes in recurrent projection layer
        bottleneck_dim: not used
    """

    def __init__(self,
                 batch_size,
                 input_size,
                 num_cell,
                 num_layers,
                 output_size,
                 parameter_init,
                 clip_gradients=None,
                 clip_activation=None,
                 dropout_ratio_input=1.0,
                 dropout_ratio_hidden=1.0,
                 num_proj=None,
                 bottleneck_dim=None):

        ctcBase.__init__(self, batch_size, input_size, num_cell, num_layers,
                         output_size, parameter_init, clip_gradients, clip_activation,
                         dropout_ratio_input, dropout_ratio_hidden)

        self.num_proj = None
        self.splice = 0

        # Define model graph
        self._build()

    def _build(self):
        """Construct network."""
        # (batch_size, max_timesteps, input_size_splice)
        inputs_shape = tf.shape(self.inputs)
        batch_size, max_timesteps = inputs_shape[0], inputs_shape[1]

        ######################################################
        # 1st conv
        # filter: (freq,time)=(3,5)
        # W: [FH, FW, InputChannel, FilterNum (OutputChannel)]
        # b: [FilterNum (OutputChannel)]
        ######################################################
        with tf.name_scope('conv1'):
            # Reshape to [batch, 40fbank + 1energy, timesteps, 3(current +
            # delta + deltadelta)]
            input_drop_rs = tf.reshape(self.inputs,
                                       # shape=[batch_size, self.input_size,
                                       # max_timesteps, 1])
                                       shape=[batch_size, int(self.input_size / 3), max_timesteps, 3])

            # Affine
            conv1_shape = [3, 5, 3, 128]  # (FH, FW, InputChannel, FilterNum)
            W_conv1 = tf.Variable(tf.truncated_normal(
                shape=conv1_shape, stddev=self.parameter_init))
            b_conv1 = tf.Variable(tf.zeros([conv1_shape[3]]))
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
            outputs = tf.nn.dropout(outputs, self.keep_prob_hidden_pl)

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
                b_conv2 = tf.Variable(tf.zeros([conv2_shape[3]]))
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
                outputs = tf.nn.dropout(outputs, self.keep_prob_hidden_pl)

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
                b_conv5 = tf.Variable(tf.zeros([conv5_shape[3]]))
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
                outputs = tf.nn.dropout(outputs, self.keep_prob_hidden_pl)

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
                W_fc11 = tf.Variable(
                    tf.truncated_normal(shape=fc_shape, stddev=self.parameter_init))
                b_fc11 = tf.Variable(tf.zeros([fc_shape[1]]))
                outputs = tf.matmul(outputs, W_fc11) + b_fc11

                # Weight decay
                # self._weight_decay(W_fc11)

                # Batch normalization
                # output = self._batch_norm(outputs)

                # Activation
                outputs = tf.nn.relu(outputs)

                # Dropout
                if i_layer != 13:
                    outputs = tf.nn.dropout(outputs, self.keep_prob_hidden_pl)

        # Reshape back to the original shape (batch_size, max_timesteps,
        # num_classes)
        outputs_3d = tf.reshape(
            outputs, shape=[batch_size, max_timesteps, self.num_classes])

        # Convert to (max_timesteps, batch_size, num_classes)
        self.logits = tf.transpose(outputs_3d, (1, 0, 2))
