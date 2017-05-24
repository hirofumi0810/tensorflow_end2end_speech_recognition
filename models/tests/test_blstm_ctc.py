#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
import functools
import unittest
import numpy as np
try:
    import tensorflow as tf
except:
    ImportError('Cannnot import tensorflow.')

sys.path.append('../')
from ctc.load_model import load
import utils


def measure_time(func):
    @functools.wraps(func)
    def _measure_time(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        elapse = time.time() - start
        print("Takes {} seconds.".format(elapse))
    return _measure_time


class TestBLSTM_CTC(unittest.TestCase):

    @measure_time
    def test_ctc(self):
        print("CTC Working check.")
        with tf.Graph().as_default() as ctc:
            self.check_training(graph=ctc, model='blstm')

    def check_training(self, graph, model):
        # load batch data
        inputs, labels, seq_len = utils.generate_batch(label_type='character')
        indices, values, dense_shape = utils.list2sparsetensor(labels)
        learning_rate = 1e-3

        # define model
        model = load(model_type='blstm_ctc')
        network = model(batch_size=1,
                        input_size=inputs[0].shape[1],
                        num_cell=256,
                        num_layers=2,
                        output_size=26,
                        clip_grad=5.0,
                        clip_activation=50,
                        dropout_ratio_input=1.0,
                        dropout_ratio_hidden=1.0,
                        num_proj=128)

        network.define()
        loss_op = network.loss()
        # train_op = network.train(optimizer='adam',
        #                          learning_rate_init=learning_rate, is_scheduled=False)
        train_op = network.train(optimizer='adam',
                                 learning_rate_init=learning_rate, is_scheduled=True)
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
            max_steps = 300
            start_time_global = time.time()
            start_time_step = time.time()
            ler_pre = 1
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
                _, loss_train = sess.run([train_op, loss_op], feed_dict=feed_dict)

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

                    # decay
                    if ler_pre == ler_train:
                        learning_rate = round(learning_rate * 0.98, 8)
                    ler_pre = ler_train

                    duration_step = time.time() - start_time_step
                    print('Step %d: loss = %.3f / ler = %.4f (%.3f sec)' %
                          (step + 1, loss_train, ler_train, duration_step))
                    start_time_step = time.time()

                    # visualize
                    labels_st = sess.run(decode_op, feed_dict=feed_dict)
                    labels_pred = utils.sparsetensor2list(labels_st, batch_size=1)
                    print(''.join(utils.num2alpha(labels_pred[0])))

            duration_global = time.time() - start_time_global
            print('Total time: %.3f sec' % (duration_global))


if __name__ == "__main__":
    unittest.main()
