#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import tensorflow as tf
# from tensorflow.python import debug as tf_debug

sys.path.append(os.path.abspath('../../'))
from models.ctc.multitask_ctc import Multitask_CTC
from models.test.util import measure_time
from models.test.data import generate_data, idx2alpha
from utils.io.labels.phone import Idx2phone
from utils.io.labels.sparsetensor import sparsetensor2list
from utils.parameter import count_total_parameters
from utils.training.learning_rate_controller import Controller


class TestMultitaskCTC(tf.test.TestCase):

    def test_multiask_ctc(self):
        print("Multitask CTC Working check.")

        # BLSTM
        self.check_training(encoder_type='multitask_blstm',
                            lstm_impl='BasicLSTMCell')
        self.check_training(encoder_type='multitask_blstm',
                            lstm_impl='LSTMCell')
        self.check_training(encoder_type='multitask_blstm',
                            lstm_impl='LSTMBlockCell')

        # LSTM
        self.check_training(encoder_type='multitask_lstm',
                            lstm_impl='BasicLSTMCell')
        self.check_training(encoder_type='multitask_lstm',
                            lstm_impl='LSTMCell')
        self.check_training(encoder_type='multitask_lstm',
                            lstm_impl='LSTMBlockCell')

    @measure_time
    def check_training(self, encoder_type, lstm_impl='LSTMBlockCell'):

        print('==================================================')
        print('  encoder_type: %s' % str(encoder_type))
        print('==================================================')

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
            model = Multitask_CTC(
                encoder_type=encoder_type,
                input_size=inputs[0].shape[1],
                num_units=256,
                num_layers_main=2,
                num_layers_sub=1,
                num_classes_main=num_classes_main,
                num_classes_sub=num_classes_sub,
                main_task_weight=0.8,
                lstm_impl=lstm_impl,
                parameter_init=0.1,
                clip_grad=5.0,
                clip_activation=50,
                num_proj=256,
                bottleneck_dim=50,
                weight_decay=1e-8)

            # Define placeholders
            model.create_placeholders()
            learning_rate_pl = tf.placeholder(tf.float32, name='learning_rate')

            # Add to the graph each operation
            loss_op, logits_main, logits_sub = model.compute_loss(
                model.inputs_pl_list[0],
                model.labels_pl_list[0],
                model.labels_sub_pl_list[0],
                model.inputs_seq_len_pl_list[0],
                model.keep_prob_input_pl_list[0],
                model.keep_prob_hidden_pl_list[0],
                model.keep_prob_output_pl_list[0])
            train_op = model.train(
                loss_op,
                optimizer='adam',
                learning_rate=learning_rate_pl)
            decode_op_main, decode_op_sub = model.decoder(
                logits_main,
                logits_sub,
                model.inputs_seq_len_pl_list[0],
                beam_width=20)
            ler_op_main, ler_op_sub = model.compute_ler(
                decode_op_main, decode_op_sub,
                model.labels_pl_list[0], model.labels_sub_pl_list[0])

            # Define learning rate controller
            learning_rate = 1e-3
            lr_controller = Controller(learning_rate_init=learning_rate,
                                       decay_start_epoch=10,
                                       decay_rate=0.98,
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
                model.inputs_pl_list[0]: inputs,
                model.labels_pl_list[0]: labels_true_char_st,
                model.labels_sub_pl_list[0]: labels_true_phone_st,
                model.inputs_seq_len_pl_list[0]: inputs_seq_len,
                model.keep_prob_input_pl_list[0]: 0.9,
                model.keep_prob_hidden_pl_list[0]: 0.9,
                model.keep_prob_output_pl_list[0]: 0.9,
                learning_rate_pl: learning_rate
            }

            idx2phone = Idx2phone(map_file_path='./phone61_ctc.txt')

            with tf.Session() as sess:
                # Initialize parameters
                sess.run(init_op)

                # Wrapper for tfdbg
                # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

                # Train model
                max_steps = 1000
                start_time_global = time.time()
                start_time_step = time.time()
                ler_train_char_pre = 1
                not_improved_count = 0
                for step in range(max_steps):

                    # Compute loss
                    _, loss_train = sess.run(
                        [train_op, loss_op], feed_dict=feed_dict)

                    # Gradient check
                    # grads = sess.run(model.clipped_grads,
                    #                  feed_dict=feed_dict)
                    # for grad in grads:
                    #     print(np.max(grad))

                    if (step + 1) % 10 == 0:
                        # Change to evaluation mode
                        feed_dict[model.keep_prob_input_pl_list[0]] = 1.0
                        feed_dict[model.keep_prob_hidden_pl_list[0]] = 1.0
                        feed_dict[model.keep_prob_output_pl_list[0]] = 1.0

                        # Compute accuracy
                        ler_train_char, ler_train_phone = sess.run(
                            [ler_op_main, ler_op_sub], feed_dict=feed_dict)

                        duration_step = time.time() - start_time_step
                        print('Step %d: loss = %.3f / cer = %.3f / per = %.3f (%.3f sec) / lr = %.5f' %
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
                        try:
                            print('  Ref: %s' %
                                  idx2alpha(labels_true_char[0]))
                            print('  Hyp: %s' %
                                  idx2alpha(labels_pred_char[0]))
                        except IndexError:
                            print('Character')
                            print('  Ref: %s' %
                                  idx2alpha(labels_true_char[0]))
                            print('  Hyp: %s' % '')

                        print('Phone')
                        try:
                            print('  Ref: %s' %
                                  idx2phone(labels_true_phone[0]))
                            print('  Hyp: %s' %
                                  idx2phone(labels_pred_phone[0]))
                        except IndexError:
                            print('  Ref: %s' %
                                  idx2phone(labels_true_phone[0]))
                            print('  Hyp: %s' % '')
                            # NOTE: This is for no prediction
                        print('---------------------------------------------' +
                              '---------------------------------------------')

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
                        feed_dict[learning_rate_pl] = learning_rate

                duration_global = time.time() - start_time_global
                print('Total time: %.3f sec' % (duration_global))


if __name__ == "__main__":
    tf.test.main()
