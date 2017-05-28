#! /usr/bin/env python
# -*- coding: utf-8 -*-

from collections import namedtuple
import tensorflow as tf
from attention_base import AttentionBase

# template
EncoderOutput = namedtuple(
    "EncoderOutput",
    "outputs final_state attention_values attention_values_length")


class BLSTMEncoder(object):
    """
    Args:
        inputs:
        seq_len:
        num_cell:
        num_layers:
        parameter_init:
        clip_activation:
        dropout_ratio_input:
        dropout_ratio_hidden:
        num_proj:
    """

    def __init__(self,
                 inputs_pl,
                 seq_len_pl,
                 keep_prob_input_pl,
                 keep_prob_hidden_pl,
                 num_cell,
                 num_layers,
                 parameter_init,
                 clip_activation,
                 num_proj):

        self.inputs_pl = inputs_pl
        self.seq_len_pl = seq_len_pl
        self.keep_prob_input_pl = keep_prob_input_pl
        self.keep_prob_hidden_pl = keep_prob_hidden_pl
        self.num_cell = num_cell
        self.num_layers = num_layers
        self.parameter_init = parameter_init
        self.clip_activation = clip_activation
        self.num_proj = num_proj

    def define(self):
        """Construct Bidirectional LSTM encoder.
        Returns:
            EncoderOutputs: tuple of
                (outputs, final_state, attention_values, attention_values_length)

                outputs:
                final_state:
                attention_values:
                attention_values_length:
        """
        with tf.name_scope('Encoder'):

            # input dropout
            input_drop = tf.nn.dropout(self.inputs_pl,
                                       self.keep_prob_input_pl,
                                       name='dropout_input')

            # hidden layers
            outputs = input_drop
            for i_layer in range(self.num_layers):
                with tf.name_scope('BiLSTM_encoder_hidden' + str(i_layer + 1)):

                    initializer = tf.random_uniform_initializer(minval=-self.parameter_init,
                                                                maxval=self.parameter_init)

                    lstm_fw = tf.contrib.rnn.LSTMCell(self.num_cell,
                                                      use_peepholes=True,
                                                      cell_clip=self.clip_activation,
                                                      initializer=initializer,
                                                      num_proj=None,
                                                      forget_bias=1.0,
                                                      state_is_tuple=True)
                    lstm_bw = tf.contrib.rnn.LSTMCell(self.num_cell,
                                                      use_peepholes=True,
                                                      cell_clip=self.clip_activation,
                                                      initializer=initializer,
                                                      num_proj=self.num_proj,
                                                      forget_bias=1.0,
                                                      state_is_tuple=True)

                    # dropout (output)
                    lstm_fw = tf.contrib.rnn.DropoutWrapper(lstm_fw,
                                                            output_keep_prob=self.keep_prob_hidden_pl)
                    lstm_bw = tf.contrib.rnn.DropoutWrapper(lstm_bw,
                                                            output_keep_prob=self.keep_prob_hidden_pl)

                    # _init_state_fw = lstm_fw.zero_state(self.batch_size, tf.float32)
                    # _init_state_bw = lstm_bw.zero_state(self.batch_size, tf.float32)
                    # initial_state_fw=_init_state_fw,
                    # initial_state_bw=_init_state_bw,

                    (outputs_fw, outputs_bw), final_states = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw=lstm_fw,
                        cell_bw=lstm_bw,
                        inputs=outputs,
                        sequence_length=self.seq_len_pl,
                        dtype=tf.float32,
                        scope='BiLSTM_' + str(i_layer + 1))

                    outputs = tf.concat(
                        axis=2, values=[outputs_fw, outputs_bw])

        return EncoderOutput(outputs=outputs,
                             final_state=final_states,
                             attention_values=outputs,
                             attention_values_length=self.seq_len_pl)
