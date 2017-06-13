#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import unittest
import tensorflow as tf

sys.path.append('../')
from attention.encoders.load_encoder import load
from util import measure_time
from data import generate_data, num2alpha, num2phone


class TestAttentionEncoder(unittest.TestCase):

    @measure_time
    def test_attention_encoder(self):
        print("Attention Encoder Working check.")
        self.check_encode(model_type='blstm_encoder', label_type='character')
        self.check_encode(model_type='lstm_encoder', label_type='character')
        self.check_encode(model_type='bgru_encoder', label_type='character')
        self.check_encode(model_type='gru_encoder', label_type='character')

    def check_encode(self, model_type, label_type):
        print('----- ' + model_type + ', ' + label_type + ' -----')
        tf.reset_default_graph()
        with tf.Graph().as_default():
            # Load batch data
            inputs, _, seq_len, target_len = generate_data(label_type=label_type,
                                                           model='attention')

            # Define model
            frame_num = inputs[0].shape[0]
            input_size = inputs[0].shape[1]
            inputs_pl = tf.placeholder(tf.float32,
                                       shape=[None, None, input_size])
            seq_len_pl = tf.placeholder(tf.int64, shape=[None])
            keep_prob_input_pl = tf.placeholder(tf.float32)
            keep_prob_hidden_pl = tf.placeholder(tf.float32)

            encoder = load(model_type)(num_unit=256,
                                       num_layer=2,
                                       keep_prob_input=keep_prob_input_pl,
                                       keep_prob_hidden=keep_prob_hidden_pl,
                                       parameter_init=0.1,
                                       clip_activation=5.0,
                                       num_proj=None)
            encoder_outputs_op = encoder(inputs=inputs_pl,
                                         inputs_seq_len=seq_len_pl)

            feed_dict = {
                encoder.inputs: inputs,
                encoder.inputs_seq_len: seq_len,
                encoder.keep_prob_input: 1.0,
                encoder.keep_prob_hidden: 1.0
            }

            # Add the variable initializer operation
            init_op = tf.global_variables_initializer()

            with tf.Session() as sess:
                # Initialize parameters
                sess.run(init_op)
                encoder_outputs = sess.run(
                    [encoder_outputs_op], feed_dict=feed_dict)
                outputs = encoder_outputs[0].outputs
                (final_state_fw,
                 final_state_bw) = encoder_outputs[0].final_state
                attention_values = encoder_outputs[0].attention_values
                attention_values_length = encoder_outputs[0].attention_values_length

                if model_type == 'blstm_encoder':
                    self.assertEqual((1, frame_num, encoder.num_unit * 2),
                                     outputs.shape)
                    self.assertEqual((1, encoder.num_unit),
                                     final_state_fw.c.shape)
                    self.assertEqual((1, encoder.num_unit),
                                     final_state_bw.c.shape)
                    self.assertEqual(
                        (1, frame_num, encoder.num_unit * 2), attention_values.shape)
                    self.assertEqual(frame_num, attention_values_length[0])

                elif model_type == 'lstm_encoder':
                    self.assertEqual((1, frame_num, encoder.num_unit),
                                     outputs.shape)
                    self.assertEqual((1, encoder.num_unit),
                                     final_state_fw.c.shape)
                    self.assertEqual((1, encoder.num_unit),
                                     final_state_bw.c.shape)
                    self.assertEqual(
                        (1, frame_num, encoder.num_unit), attention_values.shape)
                    self.assertEqual(frame_num, attention_values_length[0])

                elif model_type == 'bgru_encoder':
                    self.assertEqual((1, frame_num, encoder.num_unit * 2),
                                     outputs.shape)
                    self.assertEqual((1, encoder.num_unit),
                                     final_state_fw.shape)
                    self.assertEqual((1, encoder.num_unit),
                                     final_state_bw.shape)
                    self.assertEqual(
                        (1, frame_num, encoder.num_unit * 2), attention_values.shape)
                    self.assertEqual(frame_num, attention_values_length[0])

                elif model_type == 'gru_encoder':
                    self.assertEqual((1, frame_num, encoder.num_unit),
                                     outputs.shape)
                    self.assertEqual((1, encoder.num_unit),
                                     final_state_fw.shape)
                    self.assertEqual((1, encoder.num_unit),
                                     final_state_bw.shape)
                    print(final_state_bw.shape)
                    self.assertEqual(
                        (1, frame_num, encoder.num_unit), attention_values.shape)
                    self.assertEqual(frame_num, attention_values_length[0])


if __name__ == "__main__":
    unittest.main()
