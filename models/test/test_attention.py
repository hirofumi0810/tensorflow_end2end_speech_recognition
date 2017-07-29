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
from models.attention.blstm_attention_seq2seq import BLSTMAttetion
from models.test.util import measure_time
from models.test.data import generate_data, num2alpha, num2phone
from experiments.utils.data.sparsetensor import list2sparsetensor
from experiments.utils.parameter import count_total_parameters
from experiments.utils.training.learning_rate_controller import Controller


class TestAttention(tf.test.TestCase):

    @measure_time
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

    def check_training(self, attention_type, label_type):

        print('----- attention_type: ' + attention_type + ', label_type: ' +
              label_type + ' -----')

        tf.reset_default_graph()
        with tf.Graph().as_default():
            # Load batch data
            batch_size = 1
            inputs, labels, inputs_seq_len, labels_seq_len = generate_data(
                label_type=label_type,
                model='attention',
                batch_size=batch_size)

            # Define model graph
            num_classes = 26 + 2 if label_type == 'character' else 61 + 2
            # model = load(model_type=model_type)
            network = BLSTMAttetion(
                batch_size=batch_size,
                input_size=inputs[0].shape[1],
                encoder_num_unit=128,
                encoder_num_layer=2,
                attention_dim=64,
                attention_type=attention_type,
                decoder_num_unit=128,
                decoder_num_layer=1,
                embedding_dim=20,
                num_classes=num_classes,
                sos_index=num_classes - 2,
                eos_index=num_classes - 1,
                max_decode_length=50,
                # attention_smoothing=True,
                attention_weights_tempareture=0.5,
                logits_tempareture=1.0,
                parameter_init=0.1,
                clip_grad=5.0,
                clip_activation_encoder=50,
                clip_activation_decoder=50,
                dropout_ratio_input=0.9,
                dropout_ratio_hidden=0.9,
                dropout_ratio_output=1.0,
                weight_decay=0,
                beam_width=1,
                time_major=False)

            # Define placeholders
            network.create_placeholders(gpu_index=None)

            # Add to the graph each operation
            loss_op, logits, decoder_outputs_train, decoder_outputs_infer = network.compute_loss(
                network.inputs_pl_list[0],
                network.labels_pl_list[0],
                network.inputs_seq_len_pl_list[0],
                network.labels_seq_len_pl_list[0],
                network.keep_prob_input_pl_list[0],
                network.keep_prob_hidden_pl_list[0],
                network.keep_prob_output_pl_list[0])
            train_op = network.train(
                loss_op,
                optimizer='adam',
                learning_rate=network.learning_rate_pl_list[0])
            decode_op_train, decode_op_infer = network.decoder(
                decoder_outputs_train,
                decoder_outputs_infer)
            ler_op = network.compute_ler(network.labels_st_true_pl,
                                         network.labels_st_pred_pl)

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
                network.labels_pl_list[0]: labels,
                network.inputs_seq_len_pl_list[0]: inputs_seq_len,
                network.labels_seq_len_pl_list[0]: labels_seq_len,
                network.keep_prob_input_pl_list[0]: network.dropout_ratio_input,
                network.keep_prob_hidden_pl_list[0]: network.dropout_ratio_hidden,
                network.keep_prob_output_pl_list[0]: network.dropout_ratio_output,
                network.learning_rate_pl_list[0]: learning_rate
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
                                network.labels_st_true_pl: list2sparsetensor(
                                    labels,
                                    padded_value=27),
                                network.labels_st_pred_pl: list2sparsetensor(
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
                                  num2alpha(labels[0]))
                            print('Pred (Training) : <%s' %
                                  num2alpha(predicted_ids_train[0]))
                            print('Pred (Inference): <%s' %
                                  num2alpha(predicted_ids_infer[0]))
                        else:
                            print('True            : %s' %
                                  num2phone(labels[0], map_file_path))
                            print('Pred (Training) : < %s' %
                                  num2phone(predicted_ids_train[0], map_file_path))
                            print('Pred (Inference): < %s' %
                                  num2phone(predicted_ids_infer[0], map_file_path))

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
                        feed_dict[network.learning_rate_pl_list[0]
                                  ] = learning_rate

                duration_global = time.time() - start_time_global
                print('Total time: %.3f sec' % (duration_global))


if __name__ == "__main__":
    tf.test.main()
