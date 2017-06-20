#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate trained CTC network (CSJ corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf
import yaml

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
from data.read_dataset_ctc import DataSet
from models.ctc.load_model import load
from metric.ctc import do_eval_per, do_eval_cer


def do_eval(network, label_type, num_stack, num_skip, train_data_size,
            epoch=None):
    """Evaluate the model.
    Args:
        network: model to restore
        label_type: string, phone or character o kanji
        num_stack: int, the number of frames to stack
        num_skip: int, the number of frames to skip
        train_data_size: string, default or large
        epoch: int, the epoch to restore
    """
    # Load dataset
    eval1_data = DataSet(data_type='eval1', label_type=label_type,
                         batch_size=1,
                         train_data_size=train_data_size,
                         num_stack=num_stack, num_skip=num_skip,
                         is_sorted=False, is_progressbar=True)
    eval2_data = DataSet(data_type='eval2', label_type=label_type,
                         batch_size=1,
                         train_data_size=train_data_size,
                         num_stack=num_stack, num_skip=num_skip,
                         is_sorted=False, is_progressbar=True)
    eval3_data = DataSet(data_type='eval3', label_type=label_type,
                         batch_size=1,
                         train_data_size=train_data_size,
                         num_stack=num_stack, num_skip=num_skip,
                         is_sorted=False, is_progressbar=True)

    # Define placeholders
    network.inputs = tf.placeholder(
        tf.float32,
        shape=[None, None, network.input_size],
        name='input')
    indices_pl = tf.placeholder(tf.int64, name='indices')
    values_pl = tf.placeholder(tf.int32, name='values')
    shape_pl = tf.placeholder(tf.int64, name='shape')
    network.labels = tf.SparseTensor(indices_pl, values_pl, shape_pl)
    network.inputs_seq_len = tf.placeholder(tf.int64,
                                            shape=[None],
                                            name='inputs_seq_len')
    network.keep_prob_input = tf.placeholder(tf.float32,
                                             name='keep_prob_input')
    network.keep_prob_hidden = tf.placeholder(tf.float32,
                                              name='keep_prob_hidden')

    # Add to the graph each operation (including model definition)
    _, logits = network.compute_loss(network.inputs,
                                     network.labels,
                                     network.inputs_seq_len,
                                     network.keep_prob_input,
                                     network.keep_prob_hidden)
    decode_op = network.decoder(logits,
                                network.inputs_seq_len,
                                decode_type='beam_search',
                                beam_width=20)
    per_op = network.compute_ler(decode_op, network.labels)

    # Create a saver for writing training checkpoints
    saver = tf.train.Saver()

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(network.model_dir)

        # If check point exists
        if ckpt:
            # Use last saved model
            model_path = ckpt.model_checkpoint_path
            if epoch is not None:
                model_path = model_path.split('/')[:-1]
                model_path = '/'.join(model_path) + '/model.ckpt-' + str(epoch)
            saver.restore(sess, model_path)
            print("Model restored: " + model_path)
        else:
            raise ValueError('There are not any checkpoints.')

        if label_type in ['character', 'kanji']:
            print('=== eval1 Evaluation ===')
            cer_eval1 = do_eval_cer(
                session=sess,
                decode_op=decode_op,
                network=network,
                dataset=eval1_data,
                label_type=label_type,
                is_test=True,
                eval_batch_size=1,
                is_progressbar=True)
            print('  CER: %f %%' % (cer_eval1 * 100))

            print('=== eval2 Evaluation ===')
            cer_eval2 = do_eval_cer(
                session=sess,
                decode_op=decode_op,
                network=network,
                dataset=eval2_data,
                label_type=label_type,
                is_test=True,
                eval_batch_size=1,
                is_progressbar=True)
            print('  CER: %f %%' % (cer_eval2 * 100))

            print('=== eval3 Evaluation ===')
            cer_eval3 = do_eval_cer(
                session=sess,
                decode_op=decode_op,
                network=network,
                dataset=eval3_data,
                label_type=label_type,
                is_test=True,
                eval_batch_size=1,
                is_progressbar=True)
            print('  CER: %f %%' % (cer_eval3 * 100))

        else:
            print('=== eval1 Evaluation ===')
            per_eval1 = do_eval_per(
                session=sess,
                per_op=per_op,
                network=network,
                dataset=eval1_data,
                eval_batch_size=1,
                is_progressbar=True)
            print('  PER: %f %%' % (per_eval1 * 100))

            print('=== eval2 Evaluation ===')
            per_eval2 = do_eval_per(
                session=sess,
                per_op=per_op,
                network=network,
                dataset=eval2_data,
                eval_batch_size=1,
                is_progressbar=True)
            print('  PER: %f %%' % (per_eval2 * 100))

            print('=== eval3 Evaluation ===')
            per_eval3 = do_eval_per(
                session=sess,
                per_op=per_op,
                network=network,
                dataset=eval3_data,
                eval_batch_size=1,
                is_progressbar=True)
            print('  PER: %f %%' % (per_eval3 * 100))


def main(model_path):

    epoch = None  # if None, restore the final epoch

    # Load config file (.yml)
    with open(os.path.join(model_path, 'config.yml'), "r") as f:
        config = yaml.load(f)
        corpus = config['corpus']
        feature = config['feature']
        param = config['param']

    if corpus['label_type'] == 'phone':
        output_size = 38
    elif corpus['label_type'] == 'character':
        output_size = 147
    elif corpus['label_type'] == 'kanji':
        output_size = 3386

    # Modle setting
    CTCModel = load(model_type=config['model_name'])
    network = CTCModel(
        batch_size=param['batch_size'],
        input_size=feature['input_size'] * feature['num_stack'],
        num_unit=param['num_unit'],
        num_layer=param['num_layer'],
        bottleneck_dim=param['bottleneck_dim'],
        output_size=output_size,
        parameter_init=param['weight_init'],
        clip_grad=param['clip_grad'],
        clip_activation=param['clip_activation'],
        dropout_ratio_input=param['dropout_input'],
        dropout_ratio_hidden=param['dropout_hidden'],
        num_proj=param['num_proj'],
        weight_decay=param['weight_decay'])
    network.model_name = config['model_name']
    network.model_dir = model_path

    print(network.model_dir)
    do_eval(network=network,
            label_type=corpus['label_type'],
            num_stack=feature['num_stack'],
            num_skip=feature['num_skip'],
            train_data_size=corpus['train_data_size'],
            epoch=epoch)


if __name__ == '__main__':

    args = sys.argv
    if len(args) != 2:
        raise ValueError(
            ("Set a path to saved model.\n"
             "Usase: python eval_ctc.py path_to_saved_model"))
    main(model_path=args[1])
