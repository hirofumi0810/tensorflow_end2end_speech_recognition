#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf
# from tensorflow.python import debug as tf_debug

sys.path.append(os.path.abspath('../../'))
from models.ctc.ctc import CTC
from models.test.data import generate_data, idx2alpha
from utils.io.labels.sparsetensor import list2sparsetensor, sparsetensor2list
from utils.evaluation.edit_distance import compute_cer
from models.ctc.decoders.greedy_decoder import GreedyDecoder
from models.ctc.decoders.beam_search_decoder import BeamSearchDecoder
from utils.measure_time_func import measure_time


class TestCTCDecoder(tf.test.TestCase):

    def test(self):
        print("CTC Working check.")

        self.check(decoder_type='tf_greedy')
        self.check(decoder_type='tf_beam_search')
        self.check(decoder_type='np_greedy')
        self.check(decoder_type='np_beam_search')

    @measure_time
    def check(self, decoder_type):

        print('==================================================')
        print('  decoder_type: %s' % decoder_type)
        print('==================================================')

        tf.reset_default_graph()
        with tf.Graph().as_default():
            # Load batch data
            batch_size = 2
            num_stack = 2
            inputs, labels, inputs_seq_len = generate_data(
                label_type='character',
                model='ctc',
                batch_size=batch_size,
                num_stack=num_stack,
                splice=1)
            max_time = inputs.shape[1]

            # Define model graph
            model = CTC(encoder_type='blstm',
                        input_size=inputs[0].shape[-1],
                        splice=1,
                        num_stack=num_stack,
                        num_units=256,
                        num_layers=2,
                        num_classes=27,
                        lstm_impl='LSTMBlockCell',
                        parameter_init=0.1,
                        clip_grad_norm=5.0,
                        clip_activation=50,
                        num_proj=256,
                        weight_decay=1e-6)

            # Define placeholders
            model.create_placeholders()

            # Add to the graph each operation
            _, logits = model.compute_loss(
                model.inputs_pl_list[0],
                model.labels_pl_list[0],
                model.inputs_seq_len_pl_list[0],
                model.keep_prob_pl_list[0])
            beam_width = 20 if 'beam_search' in decoder_type else 1
            decode_op = model.decoder(logits,
                                      model.inputs_seq_len_pl_list[0],
                                      beam_width=beam_width)
            ler_op = model.compute_ler(decode_op, model.labels_pl_list[0])
            posteriors_op = model.posteriors(logits, blank_prior=1)

            if decoder_type == 'np_greedy':
                decoder = GreedyDecoder(blank_index=model.num_classes)
            elif decoder_type == 'np_beam_search':
                decoder = BeamSearchDecoder(space_index=26,
                                            blank_index=model.num_classes - 1)

            # Make feed dict
            feed_dict = {
                model.inputs_pl_list[0]: inputs,
                model.labels_pl_list[0]: list2sparsetensor(labels,
                                                           padded_value=-1),
                model.inputs_seq_len_pl_list[0]: inputs_seq_len,
                model.keep_prob_pl_list[0]: 1.0
            }

            # Create a saver for writing training checkpoints
            saver = tf.train.Saver()

            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state('./')

                # If check point exists
                if ckpt:
                    model_path = ckpt.model_checkpoint_path
                    saver.restore(sess, model_path)
                    print("Model restored: " + model_path)
                else:
                    raise ValueError('There are not any checkpoints.')

                if decoder_type in ['tf_greedy', 'tf_beam_search']:
                    # Decode
                    labels_pred_st = sess.run(decode_op, feed_dict=feed_dict)
                    labels_pred = sparsetensor2list(
                        labels_pred_st, batch_size=batch_size)

                    # Compute accuracy
                    cer = sess.run(ler_op, feed_dict=feed_dict)
                else:
                    # Compute CTC posteriors
                    probs = sess.run(posteriors_op, feed_dict=feed_dict)
                    probs = probs.reshape(-1, max_time, model.num_classes)

                    if decoder_type == 'np_greedy':
                        # Decode
                        labels_pred = decoder(probs=probs,
                                              seq_len=inputs_seq_len)

                    elif decoder_type == 'np_beam_search':
                        # Decode
                        labels_pred, scores = decoder(probs=probs,
                                                      seq_len=inputs_seq_len,
                                                      beam_width=beam_width)

                    # Compute accuracy
                    cer = compute_cer(str_pred=idx2alpha(labels_pred[0]),
                                      str_true=idx2alpha(labels[0]),
                                      normalize=True)

                # Visualize
                print('CER: %.3f %%' % (cer * 100))
                print('Ref: %s' % idx2alpha(labels[0]))
                print('Hyp: %s' % idx2alpha(labels_pred[0]))


if __name__ == "__main__":
    tf.test.main()
