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

sys.path.append(os.path.abspath('../../../'))
from experiments.timit.data.load_dataset_multitask_ctc import Dataset
from experiments.timit.visualization.core.plot.ctc import posterior_test_multitask
from models.ctc.multitask_ctc import Multitask_CTC


def do_plot(model, params, epoch=None):
    """Plot the multi-task CTC posteriors.
    Args:
        model: the model to restore
        params (dict): A dictionary of parameters
        epoch (int): the epoch to restore
    """
    # Load dataset
    test_data = Dataset(
        data_type='test', label_type_main=params['label_type_main'],
        label_type_sub=params['label_type_sub'],
        batch_size=1, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=False, progressbar=True)

    # Define placeholders
    model.create_placeholders()

    # Add to the graph each operation (including model definition)
    _, logits_main, logits_sub = model.compute_loss(
        model.inputs_pl_list[0],
        model.labels_pl_list[0],
        model.labels_sub_pl_list[0],
        model.inputs_seq_len_pl_list[0],
        model.keep_prob_input_pl_list[0],
        model.keep_prob_hidden_pl_list[0],
        model.keep_prob_output_pl_list[0])
    posteriors_op_main, posteriors_op_sub = model.posteriors(
        logits_main, logits_sub)

    # Create a saver for writing training checkpoints
    saver = tf.train.Saver()

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(model.save_path)

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
                                 model=model,
                                 dataset=test_data,
                                 label_type_main=params['label_type_main'],
                                 label_type_sub=params['label_type_sub'],
                                 save_path=model.save_path,
                                 show=False)


def main(model_path, epoch):

    # Load config file
    with open(os.path.join(model_path, 'config.yml'), "r") as f:
        config = yaml.load(f)
        params = config['param']

    # Except for a blank label
    if params['label_type_main'] == 'character':
        params['num_classes_main'] = 28
    elif params['label_type_main'] == 'character_capital_divide':
        params['num_classes_main'] = 72

    if params['label_type_sub'] == 'phone61':
        params['num_classes_sub'] = 61
    elif params['label_type_sub'] == 'phone48':
        params['num_classes_sub'] = 48
    elif params['label_type_sub'] == 'phone39':
        params['num_classes_sub'] = 39

    # Model setting
    model = Multitask_CTC(
        encoder_type=params['encoder_type'],
        input_size=params['input_size'] * params['num_stack'],
        num_units=params['num_units'],
        num_layers_main=params['num_layers_main'],
        num_layers_sub=params['num_layers_sub'],
        num_classes_main=params['num_classes_main'],
        num_classes_sub=params['num_classes_sub'],
        main_task_weight=params['main_task_weight'],
        clip_grad=params['clip_grad'],
        clip_activation=params['clip_activation'],
        num_proj=params['num_proj'],
        weight_decay=params['weight_decay'])

    model.save_path = model_path
    do_plot(model=model, params=params, epoch=epoch)


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
