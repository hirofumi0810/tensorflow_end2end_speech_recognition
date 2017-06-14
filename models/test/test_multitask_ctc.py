#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import unittest
import tensorflow as tf

sys.path.append('../')
sys.path.append('../../')
from ctc.multitask_blstm_ctc import Multitask_BLSTM_CTC
from util import measure_time
from data import generate_data, num2alpha, num2phone
from experiments.utils.sparsetensor import list2sparsetensor, sparsetensor2list


class TestCTC(unittest.TestCase):

    @measure_time
    def test_ctc(self):
        print("CTC Working check.")
        self.check_training()

    def check_training(self):
        print('----- multitask -----')
        tf.reset_default_graph()
        with tf.Graph().as_default():
            # Load batch data
            batch_size = 4
            inputs, labels_char, labels_phone, seq_len = generate_data(
                label_type='multitask',
                model='ctc',
                batch_size=batch_size)
            indices_char, values_char, dense_shape_char = list2sparsetensor(
                labels_char)
            indices_phone, values_phone, dense_shape_phone = list2sparsetensor(
                labels_phone)

            # Define model
            output_size_main = 26
            output_size_second = 61
            network = Multitask_BLSTM_CTC(
                batch_size=batch_size,
                input_size=inputs[0].shape[1],
                num_unit=256,
                num_layer_main=2,
                num_layer_second=1,
                output_size_main=output_size_main,
                output_size_second=output_size_second,
                main_task_weight=0.8,
                parameter_init=0.1,
                clip_grad=5.0,
                clip_activation=50,
                dropout_ratio_input=1.0,
                dropout_ratio_hidden=1.0,
                num_proj=None,
                weight_decay=1e-6)
            network.define()
            # NOTE: define model under tf.Graph()

            # Add to the graph each operation
            loss_op = network.compute_loss()
            learning_rate = 1e-3
            train_op = network.train(optimizer='rmsprop',
                                     learning_rate_init=learning_rate,
                                     is_scheduled=False)
            decode_op_main, decode_op_second = network.decoder(
                decode_type='beam_search',
                beam_width=20)
            ler_op_main, ler_op_second = network.compute_ler(
                decode_op_main, decode_op_second)

            # Add the variable initializer operation
            init_op = tf.global_variables_initializer()

            with tf.Session() as sess:
                # Initialize parameters
                sess.run(init_op)

                # Train model
                max_steps = 400
                start_time_global = time.time()
                start_time_step = time.time()
                ler_train_char_pre = 1
                not_improved_count = 0
                for step in range(max_steps):

                    feed_dict = {
                        network.inputs: inputs,
                        network.label_indices: indices_char,
                        network.label_values: values_char,
                        network.label_shape: dense_shape_char,
                        network.label_indices_second: indices_phone,
                        network.label_values_second: values_phone,
                        network.label_shape_second: dense_shape_phone,
                        network.seq_len: seq_len,
                        network.keep_prob_input: network.dropout_ratio_input,
                        network.keep_prob_hidden: network.dropout_ratio_hidden,
                        network.learning_rate: learning_rate
                    }

                    # Compute loss
                    _, loss_train = sess.run(
                        [train_op, loss_op], feed_dict=feed_dict)

                    # Gradient check
                    # grads = sess.run(network.clipped_grads, feed_dict=feed_dict)
                    # for grad in grads:
                    #     print(np.max(grad))

                    if (step + 1) % 10 == 0:
                        # Change feed dict for evaluation
                        feed_dict[network.keep_prob_input] = 1.0
                        feed_dict[network.keep_prob_hidden] = 1.0

                        # Compute accuracy
                        ler_train_char, ler_train_phone = sess.run(
                            [ler_op_main, ler_op_second], feed_dict=feed_dict)

                        duration_step = time.time() - start_time_step
                        print('Step %d: loss = %.3f / cer = %.4f / per = %.4f (%.3f sec)\n' %
                              (step + 1, loss_train, ler_train_char, ler_train_phone, duration_step))
                        start_time_step = time.time()

                        # Visualize
                        labels_st_char, labels_st_phone = sess.run(
                            [decode_op_main, decode_op_main], feed_dict=feed_dict)
                        labels_pred_char = sparsetensor2list(
                            labels_st_char, batch_size=1)
                        labels_pred_phone = sparsetensor2list(
                            labels_st_phone, batch_size=1)

                        # character
                        print('Character')
                        print('  True: %s' % num2alpha(labels_char[0]))
                        print('  Pred: %s' % num2alpha(labels_pred_char[0]))
                        print('Phone')
                        print('  True: %s' % num2phone(labels_phone[0]))
                        print('  Pred: %s' % num2phone(labels_pred_phone[0]))
                        print('----------------------------------------')

                        if ler_train_char >= ler_train_char_pre:
                            not_improved_count += 1
                        else:
                            not_improved_count = 0
                        if not_improved_count >= 3:
                            print('Modle is Converged.')
                            break
                        ler_train_char_pre = ler_train_char

                duration_global = time.time() - start_time_global
                print('Total time: %.3f sec' % (duration_global))


if __name__ == "__main__":
    unittest.main()
