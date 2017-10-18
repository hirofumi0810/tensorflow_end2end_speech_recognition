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
from models.attention.joint_ctc_attention import JointCTCAttention
from models.test.util import measure_time
from models.test.data import generate_data, idx2alpha
from utils.io.labels.phone import Idx2phone
from utils.io.labels.sparsetensor import list2sparsetensor
from utils.parameter import count_total_parameters
from utils.training.learning_rate_controller import Controller


class TestAttention(tf.test.TestCase):

    def test(self):
        print("Joint CTC-Attention Working check.")

        # self.check(label_type='phone')
        self.check(label_type='character')

    @measure_time
    def check(self, label_type='phone'):

        print('==================================================')
        print('  label_type: %s' % label_type)
        print('==================================================')

        tf.reset_default_graph()
        with tf.Graph().as_default():
            # Load batch data
            batch_size = 4
            inputs, labels, ctc_labels, inputs_seq_len, labels_seq_len = generate_data(
                label_type=label_type,
                model='joint_ctc_attention',
                batch_size=batch_size)

            # Define model graph
            num_classes = 27 if label_type == 'character' else 61
            model = JointCTCAttention(input_size=inputs[0].shape[1],
                                      encoder_type='blstm',
                                      encoder_num_units=256,
                                      encoder_num_layers=2,
                                      encoder_num_proj=None,
                                      attention_type='dot_product',
                                      attention_dim=128,
                                      decoder_type='lstm',
                                      decoder_num_units=256,
                                      decoder_num_layers=1,
                                      embedding_dim=64,
                                      lambda_weight=0.5,
                                      num_classes=num_classes,
                                      sos_index=num_classes,
                                      eos_index=num_classes + 1,
                                      max_decode_length=100,
                                      use_peephole=True,
                                      splice=1,
                                      parameter_init=0.1,
                                      clip_grad_norm=5.0,
                                      clip_activation_encoder=50,
                                      clip_activation_decoder=50,
                                      weight_decay=1e-8,
                                      time_major=True,
                                      sharpening_factor=1.0,
                                      logits_temperature=1.0)

            # Define placeholders
            model.create_placeholders()
            learning_rate_pl = tf.placeholder(tf.float32, name='learning_rate')

            # Add to the graph each operation
            loss_op, logits, ctc_logits, decoder_outputs_train, decoder_outputs_infer = model.compute_loss(
                model.inputs_pl_list[0],
                model.labels_pl_list[0],
                model.ctc_labels_pl_list[0],
                model.inputs_seq_len_pl_list[0],
                model.labels_seq_len_pl_list[0],
                model.keep_prob_encoder_pl_list[0],
                model.keep_prob_decoder_pl_list[0],
                model.keep_prob_embedding_pl_list[0])
            train_op = model.train(loss_op,
                                   optimizer='adam',
                                   learning_rate=learning_rate_pl)
            decode_op_train, decode_op_infer = model.decode(
                decoder_outputs_train,
                decoder_outputs_infer)
            ler_op = model.compute_ler(model.labels_st_true_pl,
                                       model.labels_st_pred_pl)

            # Define learning rate controller
            learning_rate = 1e-3
            lr_controller = Controller(learning_rate_init=learning_rate,
                                       decay_start_epoch=20,
                                       decay_rate=0.9,
                                       decay_patient_epoch=10,
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
                model.labels_pl_list[0]: labels,
                model.ctc_labels_pl_list[0]: list2sparsetensor(ctc_labels, padded_value=-1),
                model.inputs_seq_len_pl_list[0]: inputs_seq_len,
                model.labels_seq_len_pl_list[0]: labels_seq_len,
                model.keep_prob_encoder_pl_list[0]: 0.8,
                model.keep_prob_decoder_pl_list[0]: 1.0,
                model.keep_prob_embedding_pl_list[0]: 1.0,
                learning_rate_pl: learning_rate
            }

            idx2phone = Idx2phone(map_file_path='./phone61.txt')

            with tf.Session() as sess:
                # Initialize parameters
                sess.run(init_op)

                # Wrapper for tfdbg
                # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

                # Train model
                max_steps = 1000
                start_time_step = time.time()
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
                        feed_dict[model.keep_prob_encoder_pl_list[0]] = 1.0
                        feed_dict[model.keep_prob_decoder_pl_list[0]] = 1.0
                        feed_dict[model.keep_prob_embedding_pl_list[0]] = 1.0

                        # Predict class ids
                        predicted_ids_train, predicted_ids_infer = sess.run(
                            [decode_op_train, decode_op_infer],
                            feed_dict=feed_dict)

                        # Compute accuracy
                        try:
                            feed_dict_ler = {
                                model.labels_st_true_pl: list2sparsetensor(
                                    labels, padded_value=model.eos_index),
                                model.labels_st_pred_pl: list2sparsetensor(
                                    predicted_ids_infer, padded_value=model.eos_index)
                            }
                            ler_train = sess.run(
                                ler_op, feed_dict=feed_dict_ler)
                        except IndexError:
                            ler_train = 1

                        duration_step = time.time() - start_time_step
                        print('Step %d: loss = %.3f / ler = %.4f (%.3f sec) / lr = %.5f' %
                              (step + 1, loss_train, ler_train, duration_step, learning_rate))
                        start_time_step = time.time()

                        # Visualize
                        if label_type == 'character':
                            print('True            : %s' %
                                  idx2alpha(labels[0]))
                            print('Pred (Training) : <%s' %
                                  idx2alpha(predicted_ids_train[0]))
                            print('Pred (Inference): <%s' %
                                  idx2alpha(predicted_ids_infer[0]))
                        else:
                            print('True            : %s' %
                                  idx2phone(labels[0]))
                            print('Pred (Training) : < %s' %
                                  idx2phone(predicted_ids_train[0]))
                            print('Pred (Inference): < %s' %
                                  idx2phone(predicted_ids_infer[0]))

                        if ler_train < 0.1:
                            print('Model is Converged.')
                            break

                        # Update learning rate
                        learning_rate = lr_controller.decay_lr(
                            learning_rate=learning_rate,
                            epoch=step,
                            value=ler_train)
                        feed_dict[learning_rate_pl] = learning_rate


if __name__ == "__main__":
    tf.test.main()
