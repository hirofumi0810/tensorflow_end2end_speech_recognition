#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
import unittest
import tensorflow as tf

sys.path.append('../')
sys.path.append('../../')
from ctc.load_model import load
from util import measure_time
from data import generate_data, num2alpha, num2phone
from experiments.utils.data.sparsetensor import list2sparsetensor, sparsetensor2list


class TestCTC(unittest.TestCase):

    @measure_time
    def test_ctc(self):
        print("CTC Working check.")
        self.check_training(model_type='blstm_ctc', label_type='character')
        self.check_training(model_type='blstm_ctc', label_type='phone')
        self.check_training(model_type='lstm_ctc', label_type='character')
        self.check_training(model_type='lstm_ctc', label_type='phone')
        self.check_training(model_type='bgru_ctc', label_type='character')
        self.check_training(model_type='bgru_ctc', label_type='phone')
        self.check_training(model_type='gru_ctc', label_type='character')
        self.check_training(model_type='gru_ctc', label_type='phone')
        # self.check_training(model_type='cnn_ctc', label_type='phone')
        # self.check_training(model_type='cnn_ctc', label_type='phone')

    def check_training(self, model_type, label_type):
        tf.reset_default_graph()
        with tf.Graph().as_default():
            # load batch data
            inputs, labels, seq_len = generate_data(label_type=label_type,
                                                    model='ctc')
            indices, values, dense_shape = list2sparsetensor(labels)

            # define model
            output_size = 26 if label_type == 'character' else 61
            model = load(model_type=model_type)
            network = model(batch_size=1,
                            input_size=inputs[0].shape[1],
                            num_cell=256,
                            num_layers=2,
                            output_size=output_size,
                            parameter_init=0.1,
                            clip_grad=5.0,
                            clip_activation=50,
                            dropout_ratio_input=1.0,
                            dropout_ratio_hidden=1.0)

            network.define()
            loss_op = network.loss()
            learning_rate = 1e-3
            train_op = network.train(optimizer='adam',
                                     learning_rate_init=learning_rate,
                                     is_scheduled=False)

            # decode_op = network.greedy_decoder()
            decode_op = network.beam_search_decoder(beam_width=20)
            # posteriors_op = network.posteriors(decode_op)
            ler_op = network.ler(decode_op)

            # add the variable initializer operation
            init_op = tf.global_variables_initializer()

            with tf.Session() as sess:
                # initialize parameters
                sess.run(init_op)

                # train model
                max_steps = 400
                start_time_global = time.time()
                start_time_step = time.time()
                ler_train_pre = 1
                not_improved_count = 0
                for step in range(max_steps):

                    feed_dict = {
                        network.inputs_pl: inputs,
                        network.label_indices_pl: indices,
                        network.label_values_pl: values,
                        network.label_shape_pl: dense_shape,
                        network.seq_len_pl: seq_len,
                        network.keep_prob_input_pl: network.dropout_ratio_input,
                        network.keep_prob_hidden_pl: network.dropout_ratio_hidden,
                        network.lr_pl: learning_rate
                    }

                    # compute loss
                    _, loss_train = sess.run(
                        [train_op, loss_op], feed_dict=feed_dict)

                    # gradient check
                    # grads = sess.run(network.clipped_grads, feed_dict=feed_dict)
                    # for grad in grads:
                    #     print(np.max(grad))

                    if (step + 1) % 10 == 0:
                        # change feed dict for evaluation
                        feed_dict[network.keep_prob_input_pl] = 1.0
                        feed_dict[network.keep_prob_hidden_pl] = 1.0

                        # compute accuracy
                        ler_train = sess.run(ler_op, feed_dict=feed_dict)

                        duration_step = time.time() - start_time_step
                        print('Step %d: loss = %.3f / ler = %.4f (%.3f sec)' %
                              (step + 1, loss_train, ler_train, duration_step))
                        start_time_step = time.time()

                        # visualize
                        labels_st = sess.run(decode_op, feed_dict=feed_dict)
                        labels_pred = sparsetensor2list(
                            labels_st, batch_size=1)
                        if label_type == 'character':
                            print('True: %s' % num2alpha(labels[0]))
                            print('Pred: %s' % num2alpha(labels_pred[0]))
                        else:
                            print('True: %s' % num2phone(labels[0]))
                            print('Pred: %s' % num2phone(labels_pred[0]))

                        if ler_train >= ler_train_pre:
                            not_improved_count += 1
                        else:
                            not_improved_count = 0
                        if not_improved_count >= 3:
                            print('Modle is Converged.')
                            break
                        ler_train_pre = ler_train

                duration_global = time.time() - start_time_global
                print('Total time: %.3f sec' % (duration_global))


if __name__ == "__main__":
    unittest.main()
