#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot the trained Multi-task CTC posteriors (TIMIT corpus)."""

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
from data.read_dataset_multitask_ctc import DataSet
from models.ctc.load_model_multitask import load
from util import posterior_test_multitask


def do_plot(network, label_type_second, num_stack, num_skip, epoch=None):
    """Plot the Multi-task CTC posteriors.
    Args:
        network: model to restore
        label_type_second: phone39 or phone48 or phone61
        num_stack: int, the number of frames to stack
        num_skip: int, the number of frames to skip
        epoch: epoch to restore
    """
    # Load dataset
    test_data = DataSet(data_type='test', label_type_second='phone39',
                        num_stack=num_stack, num_skip=num_skip,
                        is_sorted=False, is_progressbar=True)

    # Define model
    network.define()

    # Add to the graph each operation
    decode_op_main, decode_op_second = network.decoder(
        decode_type='beam_search',
        beam_width=20)
    posteriors_op_main, posteriors_op_second = network.posteriors(
        decode_op_main, decode_op_second)

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
                                 posteriors_op_second=posteriors_op_second,
                                 network=network,
                                 dataset=test_data,
                                 label_type_second=label_type_second,
                                 save_path=network.model_dir)


def main(model_path):

    epoch = None  # if None, restore the final epoch

    # Load config file
    with open(os.path.join(model_path, 'config.yml'), "r") as f:
        config = yaml.load(f)
        corpus = config['corpus']
        feature = config['feature']
        param = config['param']

    if corpus['label_type_second'] == 'phone61':
        output_size_second = 61
    elif corpus['label_type_second'] == 'phone48':
        output_size_second = 48
    elif corpus['label_type_second'] == 'phone39':
        output_size_second = 39

    # Model setting
    CTCModel = load(model_type=config['model_name'])
    network = CTCModel(
        batch_size=1,
        input_size=feature['input_size'] * feature['num_stack'],
        num_unit=param['num_cell'],  # TODO: change to num_unit
        num_layer_main=param['num_layer_main'],
        num_layer_second=param['num_layer_second'],
        output_size_main=30,
        output_size_second=output_size_second,
        main_task_weight=param['main_task_weight'],
        clip_grad=param['clip_grad'],
        clip_activation=param['clip_activation'],
        dropout_ratio_input=param['dropout_input'],
        dropout_ratio_hidden=param['dropout_hidden'],
        num_proj=param['num_proj'],
        weight_decay=param['weight_decay'])
    network.model_name = config['model_name']
    network.model_dir = model_path

    print(network.model_dir)
    do_plot(network=network,
            label_type_second=corpus['label_type_second'],
            num_stack=feature['num_stack'],
            num_skip=feature['num_skip'],
            epoch=epoch)


if __name__ == '__main__':

    args = sys.argv
    if len(args) != 2:
        raise ValueError(
            ("Set a path to saved model.\n"
             "Usase: python restore_ctc.py path_to_saved_model"))
    main(model_path=args[1])
