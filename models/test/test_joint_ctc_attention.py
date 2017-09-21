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
from models.test.data import generate_data, idx2alpha, idx2phone
from utils.data.sparsetensor import list2sparsetensor
from utils.parameter import count_total_parameters
from utils.training.learning_rate_controller import Controller


class TestAttention(tf.test.TestCase):

    def test_attention(self):
        print("Joint CTC-Attention Working check.")
        self.check_training(label_type='phone')
        self.check_training(label_type='character')

    @measure_time
    def check_training(self, label_type):
        print('----- ' + label_type + ' -----')
        tf.reset_default_graph()
        with tf.Graph().as_default():
            # Load batch data
            batch_size = 1
            inputs, att_labels, inputs_seq_len, att_labels_seq_len, ctc_labels_st = generate_data(
                label_type=label_type,
                model='joint_ctc_attention',
                batch_size=batch_size)

            # Define model graph
            att_num_classes = 26 + 2 if label_type == 'character' else 61 + 2
            ctc_num_classes = 26 if label_type == 'character' else 61
            # model = load(model_type=model_type)
            network = JointCTCAttention(input_size=inputs[0].shape[1],
                                        encoder_num_unit=256,
                                        encoder_num_layer=2,
                                        attention_dim=128,
                                        attention_type='content',
                                        decoder_num_unit=256,
                                        decoder_num_layer=1,
                                        embedding_dim=20,
                                        att_num_classes=att_num_classes,
                                        ctc_num_classes=ctc_num_classes,
                                        att_task_weight=0.5,
                                        sos_index=att_num_classes - 2,
                                        eos_index=att_num_classes - 1,
                                        max_decode_length=50,
                                        attention_weights_tempareture=1.0,
                                        logits_tempareture=1.0,
                                        parameter_init=0.1,
                                        clip_grad=5.0,
                                        clip_activation_encoder=50,
                                        clip_activation_decoder=50,
                                        dropout_ratio_input=0.9,
                                        dropout_ratio_hidden=0.9,
                                        dropout_ratio_output=1.0,
                                        weight_decay=1e-8,
                                        beam_width=1,
                                        time_major=False)

            # Define placeholders
            network.create_placeholders()
            learning_rate_pl = tf.placeholder(tf.float32, name='learning_rate')

            # Add to the graph each operation
            loss_op, att_logits, ctc_logits, decoder_outputs_train, decoder_outputs_infer = network.compute_loss(
                network.inputs_pl_list[0],
                network.att_labels_pl_list[0],
                network.inputs_seq_len_pl_list[0],
                network.att_labels_seq_len_pl_list[0],
                network.ctc_labels_pl_list[0],
                network.keep_prob_input_pl_list[0],
                network.keep_prob_hidden_pl_list[0],
                network.keep_prob_output_pl_list[0])
            train_op = network.train(loss_op,
                                     optimizer='adam',
                                     learning_rate=learning_rate_pl)
            decode_op_train, decode_op_infer = network.decoder(
                decoder_outputs_train,
                decoder_outputs_infer)
            ler_op = network.compute_ler(network.att_labels_st_true_pl,
                                         network.att_labels_st_pred_pl)

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
                network.att_labels_pl_list[0]: att_labels,
                network.inputs_seq_len_pl_list[0]: inputs_seq_len,
                network.att_labels_seq_len_pl_list[0]: att_labels_seq_len,
                network.ctc_labels_pl_list[0]: ctc_labels_st,
                network.keep_prob_input_pl_list[0]: network.dropout_ratio_input,
                network.keep_prob_hidden_pl_list[0]: network.dropout_ratio_hidden,
                network.keep_prob_output_pl_list[0]: network.dropout_ratio_output,
                learning_rate_pl: learning_rate
            }

            map_file_path = '../../experiments/timit/metrics/mapping_files/attention/phone61_to_num.txt'

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

                        # Predict class ids
                        predicted_ids_train, predicted_ids_infer = sess.run(
                            [decode_op_train, decode_op_infer],
                            feed_dict=feed_dict)

                        # Compute accuracy
                        try:
                            feed_dict_ler = {
                                network.att_labels_st_true_pl: list2sparsetensor(
                                    att_labels,
                                    padded_value=27),
                                network.att_labels_st_pred_pl: list2sparsetensor(
                                    predicted_ids_infer,
                                    padded_value=27)
                            }
                            ler_train = sess.run(
                                ler_op, feed_dict=feed_dict_ler)
                        except ValueError:
                            ler_train = 1

                        duration_step = time.time() - start_time_step
                        print('Step %d: loss = %.3f / ler = %.4f (%.3f sec) / lr = %.5f' %
                              (step + 1, loss_train, ler_train, duration_step, learning_rate))
                        start_time_step = time.time()

                        # Visualize
                        if label_type == 'character':
                            print('True            : %s' %
                                  idx2alpha(att_labels[0]))
                            print('Pred (Training) : <%s' %
                                  idx2alpha(predicted_ids_train[0]))
                            print('Pred (Inference): <%s' %
                                  idx2alpha(predicted_ids_infer[0]))
                        else:
                            print('True            : %s' %
                                  idx2phone(att_labels[0], map_file_path))
                            print('Pred (Training) : < %s' %
                                  idx2phone(predicted_ids_train[0], map_file_path))
                            print('Pred (Inference): < %s' %
                                  idx2phone(predicted_ids_infer[0], map_file_path))

                        if ler_train >= ler_train_pre:
                            not_improved_count += 1
                        else:
                            not_improved_count = 0
                        if not_improved_count >= 10:
                            print('Model is Converged.')
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
