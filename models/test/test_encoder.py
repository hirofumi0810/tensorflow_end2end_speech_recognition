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
    def test(self):
        print("Encoder Working check.")

        # CNNs
        self.check(encoder_type='vgg_wang')
        # self.check(encoder_type='cnn_zhang')
        # self.check(encoder_type='resnet_wang')

        # BLSTM
        self.check(encoder_type='blstm', lstm_impl='BasicLSTMCell')
        self.check(encoder_type='blstm', lstm_impl='LSTMCell')
        self.check(encoder_type='blstm', lstm_impl='LSTMBlockCell')

        # LSTM
        self.check(encoder_type='lstm', lstm_impl='BasicLSTMCell')
        self.check(encoder_type='lstm', lstm_impl='LSTMCell')
        self.check(encoder_type='lstm', lstm_impl='LSTMBlockCell')

        # GRUs
        self.check(encoder_type='bgru')
        self.check(encoder_type='gru')

        # VGG-BLSTM
        self.check(encoder_type='vgg_blstm', lstm_impl='BasicLSTMCell')
        self.check(encoder_type='vgg_blstm', lstm_impl='LSTMCell')
        self.check(encoder_type='vgg_blstm', lstm_impl='LSTMBlockCell')

        # VGG-LSTM
        self.check(encoder_type='vgg_lstm', lstm_impl='BasicLSTMCell')
        self.check(encoder_type='vgg_lstm', lstm_impl='LSTMCell')
        self.check(encoder_type='vgg_lstm', lstm_impl='LSTMBlockCell')

        # Multi-task BLSTM
        self.check(encoder_type='multitask_blstm',
                   lstm_impl='BasicLSTMCell')
        self.check(encoder_type='multitask_blstm', lstm_impl='LSTMCell')
        self.check(encoder_type='multitask_blstm',
                   lstm_impl='LSTMBlockCell')

        # Multi-task LSTM
        self.check(encoder_type='multitask_lstm',
                   lstm_impl='BasicLSTMCell')
        self.check(encoder_type='multitask_lstm', lstm_impl='LSTMCell')
        self.check(encoder_type='multitask_lstm',
                   lstm_impl='LSTMBlockCell')

        # Dynamic
        # self.check(encoder_type='pyramid_blstm')
        # NOTE: this is under implementation

    def check(self, encoder_type, lstm_impl=None):

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
            inputs, _, inputs_seq_len = generate_data(
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
                    lstm_impl=lstm_impl,
                    use_peephole=True,
                    parameter_init=0.1,
                    clip_activation=5,
                    num_proj=None)
            elif encoder_type in ['bgru', 'gru']:
                encoder = load(encoder_type)(
                    num_units=256,
                    num_layers=5,
                    parameter_init=0.1)
            elif encoder_type in ['vgg_blstm', 'vgg_lstm']:
                encoder = load(encoder_type)(
                    input_size=input_size // 11,
                    splice=11,
                    num_units=256,
                    num_layers=5,
                    lstm_impl=lstm_impl,
                    use_peephole=True,
                    parameter_init=0.1,
                    clip_activation=5,
                    num_proj=None)
            elif encoder_type in ['multitask_blstm', 'multitask_lstm']:
                encoder = load(encoder_type)(
                    num_units=256,
                    num_layers_main=5,
                    num_layers_sub=3,
                    lstm_impl=lstm_impl,
                    use_peephole=True,
                    parameter_init=0.1,
                    clip_activation=5,
                    num_proj=None)
            elif encoder_type in ['vgg_wang', 'resnet_wang', 'cnn_zhang']:
                encoder = load(encoder_type)(
                    input_size=input_size // 11,
                    splice=11,
                    parameter_init=0.1)
                # NOTE: topology is pre-defined
            else:
                raise NotImplementedError

            # Create placeholders
            inputs_pl = tf.placeholder(tf.float32,
                                       shape=[None, None, input_size],
                                       name='inputs')
            inputs_seq_len_pl = tf.placeholder(tf.int32,
                                               shape=[None],
                                               name='inputs_seq_len')
            keep_prob_pl = tf.placeholder(tf.float32, name='keep_prob')

            # operation for forward computation
            if encoder_type in ['multitask_blstm', 'multitask_lstm']:
                hidden_states_op, final_state_op, hidden_states_sub_op, final_state_sub_op = encoder(
                    inputs=inputs_pl,
                    inputs_seq_len=inputs_seq_len_pl,
                    keep_prob=keep_prob_pl)
            else:
                hidden_states_op, final_state_op = encoder(
                    inputs=inputs_pl,
                    inputs_seq_len=inputs_seq_len_pl,
                    keep_prob=keep_prob_pl)

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
                inputs_seq_len_pl: inputs_seq_len,
                keep_prob_pl: 0.9
            }

            with tf.Session() as sess:
                # Initialize parameters
                sess.run(init_op)

                # Make prediction
                if encoder_type in ['multitask_blstm', 'multitask_lstm']:
                    hidden_states, final_state, hidden_states_sub, final_state_sub = sess.run(
                        [hidden_states_op, final_state_op, hidden_states_sub_op, final_state_sub_op], feed_dict=feed_dict)
                elif encoder_type in ['vgg_wang', 'resnet_wang', 'cnn_zhang']:
                    hidden_states = sess.run(
                        hidden_states_op, feed_dict=feed_dict)
                else:
                    hidden_states, final_state = sess.run(
                        [hidden_states_op, final_state_op], feed_dict=feed_dict)

                if encoder_type in ['blstm', 'bgru', 'vgg_blstm', 'multitask_blstm']:
                    self.assertEqual(
                        (batch_size, frame_num, encoder.num_units * 2), hidden_states.shape)

                    if encoder_type in ['blstm', 'vgg_blstm', 'multitask_blstm']:
                        self.assertEqual(
                            (batch_size, encoder.num_units), final_state[0].c.shape)
                        self.assertEqual(
                            (batch_size, encoder.num_units), final_state[0].h.shape)
                        self.assertEqual(
                            (batch_size, encoder.num_units), final_state[1].c.shape)
                        self.assertEqual(
                            (batch_size, encoder.num_units), final_state[1].h.shape)

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
                        self.assertEqual(
                            (batch_size, encoder.num_units), final_state[0].shape)
                        self.assertEqual(
                            (batch_size, encoder.num_units), final_state[1].shape)

                elif encoder_type in ['lstm', 'gru', 'vgg_lstm']:
                    self.assertEqual(
                        (batch_size, frame_num, encoder.num_units), hidden_states.shape)

                    if encoder_type in ['lstm', 'vgg_lstm', 'multitask_lstm']:
                        self.assertEqual(
                            (batch_size, encoder.num_units), final_state[0].c.shape)
                        self.assertEqual(
                            (batch_size, encoder.num_units), final_state[0].h.shape)

                        if encoder_type == 'multitask_lstm':
                            self.assertEqual(
                                (batch_size, frame_num, encoder.num_units), hidden_states_sub.shape)
                            self.assertEqual(
                                (batch_size, encoder.num_units), final_state_sub[0].c.shape)
                            self.assertEqual(
                                (batch_size, encoder.num_units), final_state_sub[0].h.shape)
                    else:
                        self.assertEqual(
                            (batch_size, encoder.num_units), final_state[0].shape)

                elif encoder_type in ['vgg_wang', 'resnet_wang', 'cnn_zhang']:
                    self.assertEqual(3, len(hidden_states.shape))
                    self.assertEqual(
                        (frame_num, batch_size), hidden_states.shape[:2])


if __name__ == "__main__":
    unittest.main()
