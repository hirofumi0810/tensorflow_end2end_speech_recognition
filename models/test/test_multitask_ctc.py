#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import tensorflow as tf
# from tensorflow.python import debug as tf_debug

sys.path.append('../../')
from models.ctc.load_model_multitask import load
from models.test.util import measure_time
from models.test.data import generate_data, num2alpha, num2phone
from experiments.utils.data.sparsetensor import sparsetensor2list
from experiments.utils.parameter import count_total_parameters
from experiments.utils.training.learning_rate_controller import Controller


class TestCTC(tf.test.TestCase):

    @measure_time
    def test_multiask_ctc(self):
        print("Multitask CTC Working check.")
        self.check_training(model_type='multitask_blstm_ctc')

    def check_training(self, model_type):
        print('----- model_type: %s -----' % model_type)

        tf.reset_default_graph()
        with tf.Graph().as_default():
            # Load batch data
            batch_size = 2
            inputs, labels_true_char_st, labels_true_phone_st, inputs_seq_len = generate_data(
                label_type='multitask',
                model='ctc',
                batch_size=batch_size)

            # Define model graph
            num_classes_main = 26
            num_classes_sub = 61
            model = load(model_type=model_type)
            network = model(
                batch_size=batch_size,
                input_size=inputs[0].shape[1],
                num_unit=256,
                num_layer_main=2,
                num_layer_sub=1,
                num_classes_main=num_classes_main,
                num_classes_sub=num_classes_sub,
                main_task_weight=0.8,
                parameter_init=0.1,
                clip_grad=5.0,
                clip_activation=50,
                dropout_ratio_input=0.9,
                dropout_ratio_hidden=0.9,
                dropout_ratio_output=0.9,
                num_proj=None,
                weight_decay=1e-8)

            # Define placeholders
            network.create_placeholders(gpu_index=0)

            # Add to the graph each operation
            loss_op, logits_main, logits_sub = network.compute_loss(
                network.inputs_pl_list[0],
                network.labels_pl_list[0],
                network.labels_sub_pl_list[0],
                network.inputs_seq_len_pl_list[0],
                network.keep_prob_input_pl_list[0],
                network.keep_prob_hidden_pl_list[0],
                network.keep_prob_output_pl_list[0])
            train_op = network.train(
                loss_op,
                optimizer='rmsprop',
                learning_rate=network.learning_rate_pl_list[0])
            decode_op_main, decode_op_sub = network.decoder(
                logits_main,
                logits_sub,
                network.inputs_seq_len_pl_list[0],
                decode_type='beam_search',
                beam_width=20)
            ler_op_main, ler_op_sub = network.compute_ler(
                decode_op_main, decode_op_sub,
                network.labels_pl_list[0], network.labels_sub_pl_list[0])

            # Define learning rate controller
            learning_rate = 1e-3
            lr_controller = Controller(learning_rate_init=learning_rate,
                                       decay_start_epoch=10,
                                       decay_rate=0.99,
                                       decay_patient_epoch=5,
                                       lower_better=True)

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
                network.inputs_pl_list[0]: inputs,
                network.labels_pl_list[0]: labels_true_char_st,
                network.labels_sub_pl_list[0]: labels_true_phone_st,
                network.inputs_seq_len_pl_list[0]: inputs_seq_len,
                network.keep_prob_input_pl_list[0]: network.dropout_ratio_input,
                network.keep_prob_hidden_pl_list[0]: network.dropout_ratio_hidden,
                network.keep_prob_output_pl_list[0]: network.dropout_ratio_output,
                network.learning_rate_pl_list[0]: learning_rate
            }

            map_file_path = '../../experiments/timit/metrics/mapping_files/ctc/phone61_to_num.txt'

            with tf.Session() as sess:
                # Initialize parameters
                sess.run(init_op)

                # Wrapper for tfdbg
                # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

                # Train model
                max_steps = 400
                start_time_global = time.time()
                start_time_step = time.time()
                ler_train_char_pre = 1
                not_improved_count = 0
                for step in range(max_steps):

                    # Compute loss
                    _, loss_train = sess.run(
                        [train_op, loss_op], feed_dict=feed_dict)

                    # Gradient check
                    # grads = sess.run(network.clipped_grads,
                    #                  feed_dict=feed_dict)
                    # for grad in grads:
                    #     print(np.max(grad))

                    if (step + 1) % 10 == 0:
                        # Change to evaluation mode
                        feed_dict[network.keep_prob_input_pl_list[0]] = 1.0
                        feed_dict[network.keep_prob_hidden_pl_list[0]] = 1.0
                        feed_dict[network.keep_prob_output_pl_list[0]] = 1.0

                        # Compute accuracy
                        ler_train_char, ler_train_phone = sess.run(
                            [ler_op_main, ler_op_sub], feed_dict=feed_dict)

                        duration_step = time.time() - start_time_step
                        print('Step %d: loss = %.3f / cer = %.4f / per = %.4f (%.3f sec) / lr = %.5f' %
                              (step + 1, loss_train, ler_train_char,
                               ler_train_phone, duration_step, learning_rate))
                        start_time_step = time.time()

                        # Visualize
                        labels_pred_char_st, labels_pred_phone_st = sess.run(
                            [decode_op_main, decode_op_sub],
                            feed_dict=feed_dict)
                        labels_true_char = sparsetensor2list(
                            labels_true_char_st, batch_size=batch_size)
                        labels_true_phone = sparsetensor2list(
                            labels_true_phone_st, batch_size=batch_size)
                        labels_pred_char = sparsetensor2list(
                            labels_pred_char_st, batch_size=batch_size)
                        labels_pred_phone = sparsetensor2list(
                            labels_pred_phone_st, batch_size=batch_size)
                        print('Character')
                        print('  True: %s' % num2alpha(labels_true_char[0]))
                        print('  Pred: %s' % num2alpha(labels_pred_char[0]))
                        print('Phone')
                        print('  True: %s' % num2phone(labels_true_phone[0],
                                                       map_file_path))
                        print('  Pred: %s' % num2phone(labels_pred_phone[0],
                                                       map_file_path))
                        print('----------------------------------------')

                        if ler_train_char >= ler_train_char_pre:
                            not_improved_count += 1
                        else:
                            not_improved_count = 0
                        if not_improved_count >= 5:
                            print('Modle is Converged.')
                            break
                        ler_train_char_pre = ler_train_char

                        # Update learning rate
                        learning_rate = lr_controller.decay_lr(
                            learning_rate=learning_rate,
                            epoch=step,
                            value=ler_train_char)
                        feed_dict[network.learning_rate_pl_list[0]
                                  ] = learning_rate

                duration_global = time.time() - start_time_global
                print('Total time: %.3f sec' % (duration_global))


if __name__ == "__main__":
    tf.test.main()
