#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate the trained CTC model (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf
import yaml

sys.path.append('../../../')
from experiments.timit.data.load_dataset_ctc import Dataset
from experiments.timit.metrics.ctc import do_eval_per, do_eval_cer
from models.ctc.load_model import load


def do_eval(network, param, epoch=None):
    """Evaluate the model.
    Args:
        network: model to restore
        param: A dictionary of parameters
        epoch: int, the epoch to restore
    """
    # Load dataset
    if param['label_type'] == 'character':
        test_data = Dataset(data_type='test', label_type='character',
                            batch_size=1,
                            num_stack=param['num_stack'],
                            num_skip=param['num_skip'],
                            is_sorted=False, is_progressbar=True)
    else:
        test_data = Dataset(data_type='test', label_type='phone39',
                            batch_size=1,
                            num_stack=param['num_stack'],
                            num_skip=param['num_skip'],
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

        print('Test Data Evaluation:')
        if param['label_type'] == 'character':
            cer_test = do_eval_cer(
                session=sess,
                decode_op=decode_op,
                network=network,
                dataset=test_data,
                is_progressbar=True)
            print('  CER: %f %%' % (cer_test * 100))
        else:
            per_test = do_eval_per(
                session=sess,
                decode_op=decode_op,
                per_op=per_op,
                network=network,
                dataset=test_data,
                label_type=param['label_type'],
                is_progressbar=True)
            print('  PER: %f %%' % (per_test * 100))


def main(model_path, epoch):

    # Load config file
    with open(os.path.join(model_path, 'config.yml'), "r") as f:
        config = yaml.load(f)
        param = config['param']

    # Except for a blank label
    if param['label_type'] == 'phone61':
        param['num_classes'] = 61
    elif param['label_type'] == 'phone48':
        param['num_classes'] = 48
    elif param['label_type'] == 'phone39':
        param['num_classes'] = 39
    elif param['label_type'] == 'character':
        param['num_classes'] = 33

    # Model setting
    CTCModel = load(model_type=param['model'])
    network = CTCModel(
        batch_size=1,
        input_size=param['input_size'] * param['num_stack'],
        num_unit=param['num_unit'],
        num_layer=param['num_layer'],
        num_classes=param['num_classes'],
        parameter_init=param['weight_init'],
        clip_grad=param['clip_grad'],
        clip_activation=param['clip_activation'],
        dropout_ratio_input=param['dropout_input'],
        dropout_ratio_hidden=param['dropout_hidden'],
        num_proj=param['num_proj'],
        weight_decay=param['weight_decay'])

    network.model_dir = model_path
    print(network.model_dir)
    do_eval(network=network, param=param, epoch=epoch)


if __name__ == '__main__':

    args = sys.argv
    if len(args) == 2:
        model_path = args[1]
        epoch = None
    elif len(args) == 3:
        model_path = args[1]
        epoch = args[2]
    else:
        raise ValueError(
            ("Set a path to saved model.\n"
             "Usase: python eval_ctc.py path_to_saved_model (epoch)"))
    main(model_path=model_path, epoch=epoch)
