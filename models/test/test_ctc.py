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
from models.ctc.vanilla_ctc import CTC
from models.test.util import measure_time
from models.test.data import generate_data, num2alpha
from experiments.utils.data.labels.phone import num2phone
from experiments.utils.data.sparsetensor import sparsetensor2list
from experiments.utils.parameter import count_total_parameters
from experiments.utils.training.learning_rate_controller import Controller


class TestCTC(tf.test.TestCase):

    def test_ctc(self):
        print("CTC Working check.")

        ##############################
        # BLSTM-CTC
        ##############################
        self.check_training(encoder_type='blstm', label_type='phone',
                            lstm_impl='BasicLSTMCell')
        self.check_training(encoder_type='blstm', label_type='phone',
                            lstm_impl='LSTMCell')
        self.check_training(encoder_type='blstm', label_type='phone',
                            lstm_impl='LSTMBlockCell')
        self.check_training(encoder_type='blstm', label_type='phone',
                            lstm_impl='LSTMBlockFusedCell')
        self.check_training(encoder_type='blstm', label_type='character')

        ##############################
        # LSTM-CTC
        ##############################
        self.check_training(encoder_type='lstm', label_type='phone',
                            lstm_impl='BasicLSTMCell')
        self.check_training(encoder_type='lstm', label_type='phone',
                            lstm_impl='LSTMCell')
        self.check_training(encoder_type='lstm', label_type='phone',
                            lstm_impl='LSTMBlockCell')
        self.check_training(encoder_type='lstm', label_type='phone',
                            lstm_impl='LSTMBlockFusedCell')
        self.check_training(encoder_type='lstm', label_type='character')

        ##############################
        # VGG-BLSTM-CTC
        ##############################
        self.check_training(encoder_type='vgg_blstm', label_type='phone')
        self.check_training(encoder_type='vgg_blstm', label_type='character')

        ##############################
        # VGG-LSTM-CTC
        ##############################
        self.check_training(encoder_type='vgg_lstm', label_type='phone')
        self.check_training(encoder_type='vgg_lstm', label_type='character')

        ##############################
        # BGRU-CTC
        ##############################
        self.check_training(encoder_type='bgru', label_type='phone')
        self.check_training(encoder_type='bgru', label_type='character')

        ##############################
        # GRU-CTC
        ##############################
        self.check_training(encoder_type='gru', label_type='phone')
        self.check_training(encoder_type='gru', label_type='character')

        ##############################
        # CNN-CTC
        ##############################
        # self.check_training(encoder_type='cnn', label_type='phone')
        # self.check_training(encoder_type='cnn', label_type='character')

    @measure_time
    def check_training(self, encoder_type, label_type, bidirectional=False,
                       lstm_impl='LSTMBlockCell'):

        print('==================================================')
        print('  encoder_type: %s' % encoder_type)
        print('  label_type: %s' % label_type)
        print('  lstm_impl: %s' % lstm_impl)
        print('==================================================')

        tf.reset_default_graph()
        with tf.Graph().as_default():
            # Load batch data
            batch_size = 2
            splice = 1 if encoder_type not in ['vgg_blstm', 'vgg_lstm', 'cnn'] else 11
            inputs, labels_true_st, inputs_seq_len = generate_data(
                label_type=label_type,
                model='ctc',
                batch_size=batch_size,
                splice=splice)

            # Define model graph
            num_classes = 26 if label_type == 'character' else 61
            model = CTC(encoder_type=encoder_type,
                        input_size=inputs[0].shape[-1] // splice,
                        splice=splice,
                        num_units=256,
                        num_layers=2,
                        num_classes=num_classes,
                        lstm_impl=lstm_impl,
                        parameter_init=0.1,
                        clip_grad=5.0,
                        clip_activation=50,
                        num_proj=256,
                        # bottleneck_dim=50,
                        bottleneck_dim=None,
                        weight_decay=1e-8)

            # Define placeholders
            model.create_placeholders()
            learning_rate_pl = tf.placeholder(tf.float32, name='learning_rate')

            # Add to the graph each operation
            loss_op, logits = model.compute_loss(
                model.inputs_pl_list[0],
                model.labels_pl_list[0],
                model.inputs_seq_len_pl_list[0],
                model.keep_prob_input_pl_list[0],
                model.keep_prob_hidden_pl_list[0],
                model.keep_prob_output_pl_list[0])
            train_op = model.train(loss_op,
                                   optimizer='adam',
                                   learning_rate=learning_rate_pl)
            decode_op = model.decoder(logits,
                                      model.inputs_seq_len_pl_list[0],
                                      decode_type='beam_search',
                                      beam_width=20)
            ler_op = model.compute_ler(decode_op, model.labels_pl_list[0])

            # Define learning rate controller
            learning_rate = 1e-4
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
                model.labels_pl_list[0]: labels_true_st,
                model.inputs_seq_len_pl_list[0]: inputs_seq_len,
                model.keep_prob_input_pl_list[0]: 0.9,
                model.keep_prob_hidden_pl_list[0]: 0.9,
                model.keep_prob_output_pl_list[0]: 0.9,
                learning_rate_pl: learning_rate
            }

            map_file_path = '../../experiments/timit/metrics/mapping_files/ctc/phone61_to_num.txt'

            print('aaa')
            # return 0

            with tf.Session() as sess:
                # Initialize parameters
                sess.run(init_op)

                return 0

                # Wrapper for tfdbg
                # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

                # Train model
                max_steps = 1000
                start_time_global = time.time()
                start_time_step = time.time()
                ler_train_pre = 1
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
                        ler_train = sess.run(ler_op, feed_dict=feed_dict)

                        duration_step = time.time() - start_time_step
                        print('Step %d: loss = %.3f / ler = %.4f (%.3f sec) / lr = %.5f' %
                              (step + 1, loss_train, ler_train, duration_step, learning_rate))
                        start_time_step = time.time()

                        # Visualize
                        labels_pred_st = sess.run(
                            decode_op, feed_dict=feed_dict)
                        labels_true = sparsetensor2list(
                            labels_true_st, batch_size=batch_size)

                        try:
                            labels_pred = sparsetensor2list(
                                labels_pred_st, batch_size=batch_size)
                            if label_type == 'character':
                                print('True: %s' % num2alpha(labels_true[0]))
                                print('Pred: %s' % num2alpha(labels_pred[0]))
                            else:
                                print('True: %s' % num2phone(
                                    labels_true[0], map_file_path))
                                print('Pred: %s' % num2phone(
                                    labels_pred[0], map_file_path))

                        except IndexError:
                            if label_type == 'character':
                                print('True: %s' % num2alpha(labels_true[0]))
                                print('Pred: %s' % '')
                            else:
                                print('True: %s' % num2phone(
                                    labels_true[0], map_file_path))
                                print('Pred: %s' % '')
                            # NOTE: This is for no prediction

                        if ler_train >= ler_train_pre:
                            not_improved_count += 1
                        else:
                            not_improved_count = 0
                        if ler_train < 0.05:
                            print('Modle is Converged.')
                            break
                        ler_train_pre = ler_train

                        # Update learning rate
                        learning_rate = lr_controller.decay_lr(
                            learning_rate=learning_rate,
                            epoch=step,
                            value=ler_train)
                        feed_dict[learning_rate_pl] = learning_rate

                duration_global = time.time() - start_time_global
                print('Total time: %.3f sec' % (duration_global))


if __name__ == "__main__":
    tf.test.main()
