#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import unittest
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

sys.path.append('../')
sys.path.append('../../')
from attention import blstm_attention_seq2seq
from util import measure_time
from data import generate_data, num2alpha, num2phone


class TestAttention(tf.test.TestCase):

    @measure_time
    def test_attention(self):
        print("Attention Working check.")
        self.check_training(model_type='attention', label_type='phone')
        self.check_training(model_type='attention', label_type='character')

    def check_training(self, model_type, label_type):
        print('----- ' + model_type + ', ' + label_type + ' -----')
        tf.reset_default_graph()
        with tf.Graph().as_default():
            # Load batch data
            batch_size = 4
            inputs, labels, inputs_seq_len, labels_seq_len = generate_data(
                label_type=label_type,
                model='attention',
                batch_size=batch_size)

            # Define model
            if label_type == 'character':
                output_size = 28
            else:
                output_size = 63
            # model = load(model_type=model_type)
            network = blstm_attention_seq2seq.BLSTMAttetion(
                batch_size=batch_size,
                input_size=inputs[0].shape[1],
                encoder_num_units=256,
                encoder_num_layer=2,
                attention_dim=128,
                decoder_num_units=256,
                decoder_num_layer=1,
                output_size=output_size,  # + <SOS> & <EOS>
                sos_index=output_size - 2,
                eos_index=output_size - 1,
                max_decode_length=50,
                parameter_init=0.1,
                clip_grad=5.0,
                clip_activation_encoder=50,
                clip_activation_decoder=50,
                dropout_ratio_input=1.0,
                dropout_ratio_hidden=1.0,
                weight_decay=1e-6,
                beam_width=0)

            network.define()
            # NOTE: define model under tf.Graph()

            # Add to the graph each operation
            loss_op = network.loss
            learning_rate = 1e-3
            train_op = network.train(optimizer='adam',
                                     learning_rate_init=learning_rate,
                                     is_scheduled=False)

            # decode_op = network.greedy_decoder()
            # decode_op = network.beam_search_decoder(beam_width=20)
            # posteriors_op = network.posteriors(decode_op)
            # ler_op = network.ler(decode_op)
            logits_train, predicted_ids_train, _, _, _ = network.decoder_outputs_train
            logits_train, predicted_ids_infer, _, _, _ = network.decoder_outputs_infer

            # Add the variable initializer operation
            init_op = tf.global_variables_initializer()

            with tf.Session() as sess:

                # Initialize parameters
                sess.run(init_op)

                # Wrapper for tfdbg
                # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

                # Train model
                max_steps = 400
                start_time_global = time.time()
                start_time_step = time.time()
                ler_train_pre = 1
                not_improved_count = 0
                for step in range(max_steps):

                    feed_dict = {
                        network.inputs: inputs,
                        network.labels: labels,
                        network.inputs_seq_len: inputs_seq_len,
                        network.labels_seq_len: labels_seq_len,
                        network.keep_prob_input: network.dropout_ratio_input,
                        network.keep_prob_hidden: network.dropout_ratio_hidden,
                        network.learning_rate: learning_rate
                    }

                    # Compute loss
                    _, loss_train = sess.run(
                        [train_op, loss_op],
                        feed_dict=feed_dict)

                    # Gradient check
                    # grads = sess.run(network.clipped_grads,
                    #                  feed_dict=feed_dict)
                    # for grad in grads:
                    #     print(np.max(grad))

                    if (step + 1) % 10 == 0:
                        # change feed dict for evaluation
                        feed_dict[network.keep_prob_input] = 1.0
                        feed_dict[network.keep_prob_hidden] = 1.0

                        # compute accuracy
                        # ler_train = sess.run(ler_op, feed_dict=feed_dict)

                        duration_step = time.time() - start_time_step
                        print('Step %d: loss = %.3f / ler = %.4f (%.3f sec)' %
                              (step + 1, loss_train, 1, duration_step))
                        start_time_step = time.time()

                        # Visualize
                        ids_train, ids_infer = sess.run(
                            [predicted_ids_train, predicted_ids_infer],
                            feed_dict=feed_dict)
                        if label_type == 'character':
                            print('True: %s' % num2alpha(labels[0]))
                            print('Pred: < %s' % num2alpha(ids_train[0]))
                        else:
                            print('True: %s' % num2phone(labels[0]))
                            print('Pred: < %s' % num2phone(ids_train[0]))

                        # if ler_train >= ler_train_pre:
                        #     not_improved_count += 1
                        # else:
                        #     not_improved_count = 0
                        # if not_improved_count >= 3:
                        #     print('Modle is Converged.')
                        #     break
                        # ler_train_pre = ler_train

                duration_global = time.time() - start_time_global
                print('Total time: %.3f sec' % (duration_global))


if __name__ == "__main__":
    tf.test.main()
    # unittest.main()
