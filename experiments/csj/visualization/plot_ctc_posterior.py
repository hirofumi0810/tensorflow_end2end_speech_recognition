#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot the trained CTC posteriors (CSJ corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf
import yaml

sys.path.append('../../../')
from experiments.csj.data.load_dataset_ctc import Dataset
from experiments.csj.visualization.core.plot.ctc import posterior_test
from models.ctc.load_model import load


def do_plot(network, param, epoch=None):
    """Plot the CTC posteriors.
    Args:
        network: model to restore
        param: A dictionary of parameters
        epoch: epoch to restore
    """
    # Load dataset
    eval1_data = Dataset(data_type='eval1', label_type=param['label_type'],
                         batch_size=1,
                         train_data_size=param['train_data_size'],
                         num_stack=param['num_stack'],
                         num_skip=param['num_skip'],
                         is_sorted=False, is_progressbar=True, is_gpu=False)
    eval2_data = Dataset(data_type='eval2', label_type=param['label_type'],
                         batch_size=1,
                         train_data_size=param['train_data_size'],
                         num_stack=param['num_stack'],
                         num_skip=param['num_skip'],
                         is_sorted=False, is_progressbar=True, is_gpu=False)
    eval3_data = Dataset(data_type='eval3', label_type=param['label_type'],
                         batch_size=1,
                         train_data_size=param['train_data_size'],
                         num_stack=param['num_stack'],
                         num_skip=param['num_skip'],
                         is_sorted=False, is_progressbar=True, is_gpu=False)

    # Define placeholders
    network.create_placeholders(gpu_index=None)

    # Add to the graph each operation (including model definition)
    _, logits = network.compute_loss(network.inputs_pl_list[0],
                                     network.labels_pl_list[0],
                                     network.inputs_seq_len_pl_list[0],
                                     network.keep_prob_input_pl_list[0],
                                     network.keep_prob_hidden_pl_list[0],
                                     network.keep_prob_output_pl_list[0])
    posteriors_op = network.posteriors(logits)

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

        posterior_test(session=sess,
                       posteriors_op=posteriors_op,
                       network=network,
                       dataset=eval1_data,
                       label_type=param['label_type'],
                       # save_path=network.model_dir,
                       save_path=None,
                       show=True)
        posterior_test(session=sess,
                       posteriors_op=posteriors_op,
                       network=network,
                       dataset=eval2_data,
                       label_type=param['label_type'],
                       # save_path=network.model_dir,
                       save_path=None,
                       show=True)
        posterior_test(session=sess,
                       posteriors_op=posteriors_op,
                       network=network,
                       dataset=eval3_data,
                       label_type=param['label_type'],
                       # save_path=network.model_dir,
                       save_path=None,
                       show=True)


def main(model_path, epoch):

    # Load config file
    with open(os.path.join(model_path, 'config.yml'), "r") as f:
        config = yaml.load(f)
        param = config['param']

    # Except for a blank label
    if param['label_type'] == 'kanji':
        param['num_classes'] = 3386
    elif param['label_type'] == 'kana':
        param['num_classes'] = 147
    elif param['label_type'] == 'phone':
        param['num_classes'] = 38

    # Model setting
    model = load(model_type=param['model'])
    network = model(
        batch_size=1,
        input_size=param['input_size'] * param['num_stack'],
        num_unit=param['num_unit'],
        num_layer=param['num_layer'],
        bottleneck_dim=param['bottleneck_dim'],
        num_classes=param['num_classes'],
        parameter_init=param['weight_init'],
        clip_grad=param['clip_grad'],
        clip_activation=param['clip_activation'],
        dropout_ratio_input=param['dropout_input'],
        dropout_ratio_hidden=param['dropout_hidden'],
        num_proj=param['num_proj'],
        weight_decay=param['weight_decay'])

    network.model_dir = model_path
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
             "Usase: python plot_ctc_posterior.py path_to_saved_model"))
    main(model_path=model_path, epoch=epoch)
