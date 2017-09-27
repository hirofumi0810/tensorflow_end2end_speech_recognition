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
from models.attention.blstm_attention_seq2seq import AttentionSeq2Seq
from models.test.util import measure_time
from models.test.data import generate_data, idx2alpha
from utils.io.labels.phone import idx2phone
from utils.io.labels.sparsetensor import list2sparsetensor
from utils.parameter import count_total_parameters
from utils.training.learning_rate_controller import Controller


class TestAttention(tf.test.TestCase):

    def test_attention(self):
        print("Attention Working check.")
        self.check_training(attention_type='hybrid', label_type='phone')
        self.check_training(attention_type='hybrid', label_type='character')

        self.check_training(attention_type='location', label_type='phone')
        self.check_training(attention_type='location', label_type='character')

        self.check_training(attention_type='content', label_type='phone')
        self.check_training(attention_type='content', label_type='character')

        self.check_training(attention_type='layer_dot', label_type='phone')
        self.check_training(attention_type='layer_dot', label_type='character')

    @measure_time
    def check_training(self, attention_type, label_type):

        print('==================================================')
        print('  attention_type: %s' % attention_type)
        print('  label_type: %s' % label_type)
        print('==================================================')
        # encoder_type, decoder_type

        tf.reset_default_graph()
        with tf.Graph().as_default():
            # Load batch data
            batch_size = 1
            inputs, labels, inputs_seq_len, labels_seq_len = generate_data(
                label_type=label_type,
                model='attention',
                batch_size=batch_size)

            # Define model graph
            num_classes = 28 if label_type == 'character' else 63
            model = AttentionSeq2Seq(input_size=inputs[0].shape[1],
                                     encoder_type='blstm',
                                     encoder_num_units=256,
                                     encoder_num_layers=2,
                                     attention_dim=64,
                                     attention_type=attention_type,
                                     decoder_type='lstm',
                                     decoder_num_units=256,
                                     decoder_num_layers=1,
                                     embedding_dim=20,
                                     num_classes=num_classes,
                                     sos_index=num_classes - 2,
                                     eos_index=num_classes - 1,
                                     max_decode_length=50,
                                     attention_smoothing=False,
                                     attention_weights_tempareture=1.0,
                                     logits_tempareture=1.0,
                                     parameter_init=0.1,
                                     clip_grad=5.0,
                                     clip_activation_encoder=50,
                                     clip_activation_decoder=50,
                                     weight_decay=1e-8,
                                     beam_width=1,
                                     time_major=False)

            # Define placeholders
            model.create_placeholders()
            learning_rate_pl = tf.placeholder(tf.float32, name='learning_rate')

            # Add to the graph each operation
            loss_op, logits, decoder_outputs_train, decoder_outputs_infer = model.compute_loss(
                model.inputs_pl_list[0],
                model.labels_pl_list[0],
                model.inputs_seq_len_pl_list[0],
                model.labels_seq_len_pl_list[0],
                model.keep_prob_input_pl_list[0],
                model.keep_prob_hidden_pl_list[0],
                model.keep_prob_output_pl_list[0])
            train_op = model.train(loss_op,
                                   optimizer='adam',
                                   learning_rate=learning_rate_pl)
            decode_op_train, decode_op_infer = model.decoder(
                decoder_outputs_train, decoder_outputs_infer)
            ler_op = model.compute_ler(
                model.labels_st_true_pl, model.labels_st_pred_pl)

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
                model.labels_pl_list[0]: labels,
                model.inputs_seq_len_pl_list[0]: inputs_seq_len,
                model.labels_seq_len_pl_list[0]: labels_seq_len,
                model.keep_prob_input_pl_list[0]: 0.9,
                model.keep_prob_hidden_pl_list[0]: 0.9,
                model.keep_prob_output_pl_list[0]: 1.0,
                learning_rate_pl: learning_rate
            }

            map_file_path = '../../experiments/timit/metrics/mapping_files/attention/phone61.txt'

            with tf.Session() as sess:

                # Initialize parameters
                sess.run(init_op)

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

                        # Predict class ids
                        predicted_ids_train, predicted_ids_infer = sess.run(
                            [decode_op_train, decode_op_infer],
                            feed_dict=feed_dict)

                        # Compute accuracy
                        try:
                            feed_dict_ler = {
                                model.labels_st_true_pl: list2sparsetensor(
                                    labels, padded_value=27),
                                model.labels_st_pred_pl: list2sparsetensor(
                                    predicted_ids_infer, padded_value=27)
                            }
                            ler_train = sess.run(ler_op, feed_dict=feed_dict_ler)

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
                                  idx2phone(labels[0], map_file_path))
                            print('Pred (Training) : < %s' %
                                  idx2phone(predicted_ids_train[0], map_file_path))
                            print('Pred (Inference): < %s' %
                                  idx2phone(predicted_ids_infer[0], map_file_path))

                        if ler_train >= ler_train_pre:
                            not_improved_count += 1
                        else:
                            not_improved_count = 0
                        if ler_train < 0.05:
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
