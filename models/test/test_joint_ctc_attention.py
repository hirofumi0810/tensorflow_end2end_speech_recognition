#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import tensorflow as tf
from tensorflow.python import debug as tf_debug

sys.path.append('../../')
from models.attention.joint_ctc_attention import JointCTCAttention
from models.test.util import measure_time
from models.test.data import generate_data, num2alpha, num2phone
from experiments.utils.sparsetensor import list2sparsetensor
from experiments.utils.parameter import count_total_parameters


class TestAttention(tf.test.TestCase):

    @measure_time
    def test_attention(self):
        print("Joint CTC-Attention Working check.")
        # self.check_training(label_type='phone')
        self.check_training(label_type='character')

    def check_training(self, label_type):
        print('----- ' + label_type + ' -----')
        tf.reset_default_graph()
        with tf.Graph().as_default():
            # Load batch data
            batch_size = 4
            inputs, att_labels, inputs_seq_len, att_labels_seq_len, ctc_labels_st = generate_data(
                label_type=label_type,
                model='joint_ctc_attention',
                batch_size=batch_size)

            # Define placeholders
            inputs_pl = tf.placeholder(tf.float32,
                                       shape=[batch_size, None,
                                              inputs.shape[-1]],
                                       name='inputs')

            # `[batch_size, max_time]`
            att_labels_pl = tf.placeholder(tf.int32,
                                           shape=[None, None],
                                           name='att_labels')
            indices_pl = tf.placeholder(tf.int64, name='indices')
            values_pl = tf.placeholder(tf.int32, name='values')
            shape_pl = tf.placeholder(tf.int64, name='shape')
            ctc_labels_pl = tf.SparseTensor(indices_pl, values_pl, shape_pl)

            # These are prepared for computing LER of attention outputs
            indices_true_pl = tf.placeholder(tf.int64, name='indices')
            values_true_pl = tf.placeholder(tf.int32, name='values')
            shape_true_pl = tf.placeholder(tf.int64, name='shape')
            att_labels_st_true_pl = tf.SparseTensor(indices_true_pl,
                                                    values_true_pl,
                                                    shape_true_pl)
            indices_pred_pl = tf.placeholder(tf.int64, name='indices')
            values_pred_pl = tf.placeholder(tf.int32, name='values')
            shape_pred_pl = tf.placeholder(tf.int64, name='shape')
            att_labels_st_pred_pl = tf.SparseTensor(indices_pred_pl,
                                                    values_pred_pl,
                                                    shape_pred_pl)
            inputs_seq_len_pl = tf.placeholder(tf.int32,
                                               shape=[None],
                                               name='inputs_seq_len')
            att_labels_seq_len_pl = tf.placeholder(tf.int32,
                                                   shape=[None],
                                                   name='att_labels_seq_len')
            keep_prob_input_pl = tf.placeholder(tf.float32,
                                                name='keep_prob_input')
            keep_prob_hidden_pl = tf.placeholder(tf.float32,
                                                 name='keep_prob_hidden')

            # Define model graph
            att_num_classes = 26 + 2 if label_type == 'character' else 61 + 2
            ctc_num_classes = 26 if label_type == 'character' else 61
            # model = load(model_type=model_type)
            network = JointCTCAttention(
                batch_size=batch_size,
                input_size=inputs[0].shape[1],
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
                dropout_ratio_input=1.0,
                dropout_ratio_hidden=1.0,
                weight_decay=0,
                beam_width=0,
                time_major=False)

            # Add to the graph each operation
            loss_op, att_logits, ctc_logits, decoder_outputs_train, decoder_outputs_infer = network.compute_loss(
                inputs_pl,
                att_labels_pl,
                inputs_seq_len_pl,
                att_labels_seq_len_pl,
                ctc_labels_pl,
                keep_prob_input_pl,
                keep_prob_hidden_pl)
            learning_rate = 1e-3
            train_op = network.train(loss_op,
                                     optimizer='rmsprop',
                                     learning_rate_init=learning_rate,
                                     is_scheduled=False)
            decode_op_train, decode_op_infer = network.decoder(
                decoder_outputs_train,
                decoder_outputs_infer,
                decode_type='greedy',
                beam_width=1)
            ler_op = network.compute_ler(att_labels_st_true_pl,
                                         att_labels_st_pred_pl)

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
                inputs_pl: inputs,
                att_labels_pl: att_labels,
                inputs_seq_len_pl: inputs_seq_len,
                att_labels_seq_len_pl: att_labels_seq_len,
                ctc_labels_pl: ctc_labels_st,
                keep_prob_input_pl: network.dropout_ratio_input,
                keep_prob_hidden_pl: network.dropout_ratio_hidden,
                network.lr: learning_rate
            }

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
                        feed_dict[keep_prob_input_pl] = 1.0
                        feed_dict[keep_prob_hidden_pl] = 1.0

                        # Predict class ids
                        predicted_ids_train, predicted_ids_infer = sess.run(
                            [decode_op_train, decode_op_infer],
                            feed_dict=feed_dict)

                        # Compute accuracy
                        feed_dict_ler = {
                            att_labels_st_true_pl: list2sparsetensor(att_labels),
                            att_labels_st_pred_pl: list2sparsetensor(predicted_ids_infer)
                        }
                        ler_train = sess.run(ler_op, feed_dict=feed_dict_ler)

                        duration_step = time.time() - start_time_step
                        print('Step %d: loss = %.3f / ler = %.4f (%.3f sec)' %
                              (step + 1, loss_train, ler_train, duration_step))
                        start_time_step = time.time()

                        # Visualize
                        if label_type == 'character':
                            print('True            : %s' %
                                  num2alpha(att_labels[0]))
                            print('Pred (Training) : <%s' %
                                  num2alpha(predicted_ids_train[0]))
                            print('Pred (Inference): <%s' %
                                  num2alpha(predicted_ids_infer[0]))
                        else:
                            print('True            : %s' %
                                  num2phone(att_labels[0]))
                            print('Pred (Training) : < %s' %
                                  num2phone(predicted_ids_train[0]))
                            print('Pred (Inference): < %s' %
                                  num2phone(predicted_ids_infer[0]))

                        if ler_train >= ler_train_pre:
                            not_improved_count += 1
                        else:
                            not_improved_count = 0
                        if not_improved_count >= 10:
                            print('Model is Converged.')
                            break
                        ler_train_pre = ler_train

                duration_global = time.time() - start_time_global
                print('Total time: %.3f sec' % (duration_global))


if __name__ == "__main__":
    tf.test.main()
