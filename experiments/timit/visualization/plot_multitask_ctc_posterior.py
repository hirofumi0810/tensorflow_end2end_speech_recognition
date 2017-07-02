#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot the trained multi-task CTC posteriors (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf
import yaml

sys.path.append('../../../')
from experiments.timit.data.load_dataset_multitask_ctc import Dataset
from experiments.timit.visualization.util_plot_ctc import posterior_test_multitask
from models.ctc.load_model_multitask import load


def do_plot(network, param, epoch=None):
    """Plot the multi-task CTC posteriors.
    Args:
        network: model to restore
        param: A dictionary of parameters
        epoch: int, the epoch to restore
    """
    # Load dataset
    test_data = Dataset(data_type='test',
                        label_type_main='character',
                        label_type_sub=param['label_type_sub'],
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
    indices_sub_pl = tf.placeholder(tf.int64, name='indices_sub')
    values_sub_pl = tf.placeholder(tf.int32, name='values_sub')
    shape_sub_pl = tf.placeholder(tf.int64, name='shape_sub')
    network.labels_sub = tf.SparseTensor(indices_sub_pl,
                                         values_sub_pl,
                                         shape_sub_pl)
    network.inputs_seq_len = tf.placeholder(tf.int64,
                                            shape=[None],
                                            name='inputs_seq_len')
    network.keep_prob_input = tf.placeholder(tf.float32,
                                             name='keep_prob_input')
    network.keep_prob_hidden = tf.placeholder(tf.float32,
                                              name='keep_prob_hidden')

    # Add to the graph each operation (including model definition)
    _, logits_main, logits_sub = network.compute_loss(
        network.inputs,
        network.labels,
        network.labels_sub,
        network.inputs_seq_len,
        network.keep_prob_input,
        network.keep_prob_hidden)
    posteriors_op_main, posteriors_op_sub = network.posteriors(
        logits_main, logits_sub)

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

        # Visualize
        posterior_test_multitask(session=sess,
                                 posteriors_op_main=posteriors_op_main,
                                 posteriors_op_sub=posteriors_op_sub,
                                 network=network,
                                 dataset=test_data,
                                 label_type_sub=param['label_type_sub'],
                                 save_path=network.model_dir,
                                 show=False)


def main(model_path, epoch):

    # Load config file
    with open(os.path.join(model_path, 'config.yml'), "r") as f:
        config = yaml.load(f)
        param = config['param']

    # Except for a blank label
    if param['label_type_sub'] == 'phone61':
        param['num_classes_sub'] = 61
    elif param['label_type_sub'] == 'phone48':
        param['num_classes_sub'] = 48
    elif param['label_type_sub'] == 'phone39':
        param['num_classes_sub'] = 39

    # Model setting
    CTCModel = load(model_type=config['model_name'])
    network = CTCModel(
        batch_size=1,
        input_size=param['input_size'] * param['num_stack'],
        num_unit=param['num_unit'],
        num_layer_main=param['num_layer_main'],
        num_layer_sub=param['num_layer_sub'],
        num_classes_main=30,
        num_classes_sub=param['num_classes_sub'],
        main_task_weight=param['main_task_weight'],
        clip_grad=param['clip_grad'],
        clip_activation=param['clip_activation'],
        dropout_ratio_input=param['dropout_input'],
        dropout_ratio_hidden=param['dropout_hidden'],
        num_proj=param['num_proj'],
        weight_decay=param['weight_decay'])

    network.model_dir = model_path
    print(network.model_dir)
    do_plot(network=network, param=param, epoch=epoch)


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
             "Usase: python plot_multitask_ctc_posterior.py path_to_saved_model"))
    main(model_path=model_path, epoch=epoch)
