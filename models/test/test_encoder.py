#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import unittest
import tensorflow as tf

sys.path.append(os.path.abspath('../../'))
from models.encoders.load_encoder import load
from models.test.util import measure_time
from models.test.data import generate_data
from utils.parameter import count_total_parameters


class TestEncoder(unittest.TestCase):

    @measure_time
    def test_attention_encoder(self):

        print("Encoder Working check.")

        # BLSTM
        self.check_encode(encoder_type='blstm', lstm_impl='BasicLSTMCell')
        self.check_encode(encoder_type='blstm', lstm_impl='LSTMCell')
        self.check_encode(encoder_type='blstm', lstm_impl='LSTMBlockCell')
        # self.check_encode(encoder_type='blstm', lstm_impl='LSTMBlockFusedCell')
        # self.check_encode(encoder_type='blstm', lstm_impl='CudnnLSTM')

        # LSTM
        self.check_encode(encoder_type='lstm', lstm_impl='BasicLSTMCell')
        self.check_encode(encoder_type='lstm', lstm_impl='LSTMCell')
        self.check_encode(encoder_type='lstm', lstm_impl='LSTMBlockCell')
        # self.check_encode(encoder_type='lstm', lstm_impl='LSTMBlockFusedCell')
        # self.check_encode(encoder_type='lstm', lstm_impl='CudnnLSTM')

        # GRUs
        self.check_encode(encoder_type='bgru')
        self.check_encode(encoder_type='gru')

        # VGG-BLSTM
        self.check_encode(encoder_type='vgg_blstm', lstm_impl='BasicLSTMCell')
        self.check_encode(encoder_type='vgg_blstm', lstm_impl='LSTMCell')
        self.check_encode(encoder_type='vgg_blstm', lstm_impl='LSTMBlockCell')

        # VGG-LSTM
        self.check_encode(encoder_type='vgg_lstm', lstm_impl='BasicLSTMCell')
        self.check_encode(encoder_type='vgg_lstm', lstm_impl='LSTMCell')
        self.check_encode(encoder_type='vgg_lstm', lstm_impl='LSTMBlockCell')

        # CNNs
        self.check_encode(encoder_type='vgg_wang')
        self.check_encode(encoder_type='cnn_zhang')
        # self.check_encode(encoder_type='resnet_wang')

        # Multi-task
        self.check_encode(encoder_type='multitask_blstm', lstm_impl='BasicLSTMCell')
        self.check_encode(encoder_type='multitask_blstm', lstm_impl='LSTMCell')
        self.check_encode(encoder_type='multitask_blstm', lstm_impl='LSTMBlockCell')
        self.check_encode(encoder_type='multitask_lstm', lstm_impl='BasicLSTMCell')
        self.check_encode(encoder_type='multitask_lstm', lstm_impl='LSTMCell')
        self.check_encode(encoder_type='multitask_lstm', lstm_impl='LSTMBlockCell')

        # Dynamic
        # self.check_encode(encoder_type='pyramidal_blstm')

    def check_encode(self, encoder_type, lstm_impl=None):

        print('==================================================')
        print('  encoder_type: %s' % encoder_type)
        print('  lstm_impl: %s' % lstm_impl)
        print('==================================================')

        tf.reset_default_graph()
        with tf.Graph().as_default():
            # Load batch data
            batch_size = 4
            splice = 11 if encoder_type in ['vgg_blstm', 'vgg_lstm', 'vgg_wang',
                                            'resnet_wang', 'cnn_zhang'] else 1
            inputs, _, _ = generate_data(
                label_type='character',
                model='ctc',
                batch_size=batch_size,
                splice=splice)
            frame_num, input_size = inputs[0].shape

            # Define model graph
            if encoder_type in ['blstm', 'lstm']:
                encoder = load(encoder_type)(
                    num_units=256,
                    num_layers=5,
                    num_classes=0,  # return hidden states
                    lstm_impl=lstm_impl,
                    parameter_init=0.1)
            elif encoder_type in ['bgru', 'gru']:
                encoder = load(encoder_type)(
                    num_units=256,
                    num_layers=5,
                    num_classes=0,  # return hidden states
                    parameter_init=0.1)
            elif encoder_type in ['vgg_blstm', 'vgg_lstm']:
                encoder = load(encoder_type)(
                    input_size=input_size // 11,
                    splice=11,
                    num_units=256,
                    num_layers=5,
                    num_classes=0,  # return hidden states
                    lstm_impl=lstm_impl,
                    parameter_init=0.1)
            elif encoder_type in ['multitask_blstm', 'multitask_lstm']:
                encoder = load(encoder_type)(
                    num_units=256,
                    num_layers_main=5,
                    num_layers_sub=3,
                    num_classes_main=0,  # return hidden states
                    num_classes_sub=0,  # return hidden states
                    lstm_impl=lstm_impl,
                    parameter_init=0.1)
            elif encoder_type in ['vgg_wang', 'resnet_wang', 'cnn_zhang']:
                encoder = load(encoder_type)(
                    input_size=input_size // 11,
                    splice=11,
                    num_classes=27,
                    parameter_init=0.1)
                # NOTE: topology is pre-defined
            else:
                raise NotImplementedError

            # Create placeholders
            inputs_pl = tf.placeholder(tf.float32,
                                       shape=[None, None, input_size],
                                       name='inputs')
            keep_prob_input_pl = tf.placeholder(tf.float32,
                                                name='keep_prob_input')
            keep_prob_hidden_pl = tf.placeholder(tf.float32,
                                                 name='keep_prob_hidden')
            keep_prob_output_pl = tf.placeholder(tf.float32,
                                                 name='keep_prob_output')

            # operation for forward computation
            if encoder_type in ['multitask_blstm', 'multitask_lstm']:
                hidden_states_op, final_state_op, hidden_states_sub_op, final_state_sub_op = encoder(
                    inputs=inputs_pl,
                    keep_prob_input=keep_prob_input_pl,
                    keep_prob_hidden=keep_prob_hidden_pl,
                    keep_prob_output=keep_prob_output_pl)
            else:
                hidden_states_op, final_state_op = encoder(
                    inputs=inputs_pl,
                    keep_prob_input=keep_prob_input_pl,
                    keep_prob_hidden=keep_prob_hidden_pl,
                    keep_prob_output=keep_prob_output_pl)

            # Add the variable initializer operation
            init_op = tf.global_variables_initializer()

            # Count total parameters
            parameters_dict, total_parameters = count_total_parameters(
                tf.trainable_variables())
            for parameter_name in sorted(parameters_dict.keys()):
                print("%s %d" %
                      (parameter_name, parameters_dict[parameter_name]))
            print("Total %d variables, %s M parameters" %
                  (len(parameters_dict.keys()),
                   "{:,}".format(total_parameters / 1000000)))

            # Make feed dict
            feed_dict = {
                inputs_pl: inputs,
                # inputs_seq_len_pl: inputs_seq_len,
                keep_prob_input_pl: 0.9,
                keep_prob_hidden_pl: 0.9,
                keep_prob_output_pl: 1.0
            }

            with tf.Session() as sess:
                # Initialize parameters
                sess.run(init_op)

                # Make prediction
                if encoder_type in ['multitask_blstm', 'multitask_lstm']:
                    hidden_states, final_state, hidden_states_sub, final_state_sub = sess.run(
                        [hidden_states_op, final_state_op, hidden_states_sub_op, final_state_sub_op], feed_dict=feed_dict)
                elif encoder_type in ['vgg_wang', 'resnet_wang', 'cnn_zhang']:
                    hidden_states = sess.run(hidden_states_op, feed_dict=feed_dict)
                else:
                    hidden_states, final_state = sess.run(
                        [hidden_states_op, final_state_op], feed_dict=feed_dict)

                if encoder_type in ['blstm', 'bgru', 'vgg_blstm', 'multitask_blstm']:
                    self.assertEqual(
                        (batch_size, frame_num, encoder.num_units * 2), hidden_states.shape)

                    if encoder_type in ['blstm', 'vgg_blstm', 'multitask_blstm']:
                        self.assertEqual((batch_size, encoder.num_units), final_state[0].c.shape)
                        self.assertEqual((batch_size, encoder.num_units), final_state[0].h.shape)
                        self.assertEqual((batch_size, encoder.num_units), final_state[1].c.shape)
                        self.assertEqual((batch_size, encoder.num_units), final_state[1].h.shape)

                        if encoder_type == 'multitask_blstm':
                            self.assertEqual(
                                (batch_size, frame_num, encoder.num_units * 2), hidden_states_sub.shape)
                            self.assertEqual(
                                (batch_size, encoder.num_units), final_state_sub[0].c.shape)
                            self.assertEqual(
                                (batch_size, encoder.num_units), final_state_sub[0].h.shape)
                            self.assertEqual(
                                (batch_size, encoder.num_units), final_state_sub[1].c.shape)
                            self.assertEqual(
                                (batch_size, encoder.num_units), final_state_sub[1].h.shape)
                    else:
                        self.assertEqual((batch_size, encoder.num_units), final_state[0].shape)
                        self.assertEqual((batch_size, encoder.num_units), final_state[1].shape)

                elif encoder_type in ['lstm', 'gru', 'vgg_lstm']:
                    self.assertEqual(
                        (batch_size, frame_num, encoder.num_units), hidden_states.shape)

                    if encoder_type in ['lstm', 'vgg_lstm', 'multitask_lstm']:
                        self.assertEqual((batch_size, encoder.num_units), final_state[0].c.shape)
                        self.assertEqual((batch_size, encoder.num_units), final_state[0].h.shape)

                        if encoder_type == 'multitask_lstm':
                            self.assertEqual(
                                (batch_size, frame_num, encoder.num_units), hidden_states_sub.shape)
                            self.assertEqual(
                                (batch_size, encoder.num_units), final_state_sub[0].c.shape)
                            self.assertEqual(
                                (batch_size, encoder.num_units), final_state_sub[0].h.shape)
                    else:
                        self.assertEqual((batch_size, encoder.num_units), final_state[0].shape)

                elif encoder_type in ['vgg_wang', 'resnet_wang', 'cnn_zhang']:
                    self.assertEqual(
                        (frame_num, batch_size, encoder.num_classes), hidden_states.shape)


if __name__ == "__main__":
    unittest.main()
