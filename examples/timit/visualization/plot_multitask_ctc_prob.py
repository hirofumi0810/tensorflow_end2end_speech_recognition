#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot the trained multi-task CTC posteriors (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath, isdir
import sys
import numpy as np
import tensorflow as tf
import yaml
import argparse

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
sns.set_style("white")
blue = '#4682B4'
orange = '#D2691E'
green = '#006400'

sys.path.append(abspath('../../../'))
from experiments.timit.data.load_dataset_multitask_ctc import Dataset
from experiments.timit.visualization.core.plot.ctc import plot
from models.ctc.multitask_ctc import MultitaskCTC
from utils.directory import mkdir_join

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--model_path', type=str,
                    help='path to the model to evaluate')
parser.add_argument('--eval_batch_size', type=str, default=1,
                    help='the size of mini-batch in evaluation')


def do_plot(model, params, epoch, eval_batch_size):
    """Plot the multi-task CTC posteriors.
    Args:
        model: the model to restore
        params (dict): A dictionary of parameters
        epoch (int): the epoch to restore
        eval_batch_size (int): the size of mini-batch in evaluation
    """
    # Load dataset
    test_data = Dataset(
        data_type='test', label_type_main=params['label_type_main'],
        label_type_sub=params['label_type_sub'],
        batch_size=eval_batch_size, splice=params['splice'],
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
        model.keep_prob_pl_list[0])
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
            if epoch != -1:
                model_path = model_path.split('/')[:-1]
                model_path = '/'.join(model_path) + '/model.ckpt-' + str(epoch)
            saver.restore(sess, model_path)
            print("Model restored: " + model_path)
        else:
            raise ValueError('There are not any checkpoints.')

        plot(session=sess,
             posteriors_op_main=posteriors_op_main,
             posteriors_op_sub=posteriors_op_sub,
             model=model,
             dataset=test_data,
             label_type_main=params['label_type_main'],
             label_type_sub=params['label_type_sub'],
             num_stack=params['num_stack'],
             save_path=mkdir_join(save_path, 'ctc_output'),
             show=False)


def plot(session, posteriors_op_main, posteriors_op_sub,
         model, dataset, label_type_main, label_type_sub,
         num_stack=1, save_path=None, show=False):
    """Visualize label posteriors of the multi-task CTC model.
    Args:
        session: session of training model
        posteriois_op_main: operation for computing posteriors in the main task
        posteriois_op_sub: operation for computing posteriors in the sub task
        model: model to evaluate
        dataset: An instance of a `Dataset` class
        label_type_main (string): character or character_capital_divide
        label_type_sub (string): phone39 or phone48 or phone61
        num_stack (int): the number of frames to stack
        save_path (string): path to save ctc outpus
        show (bool, optional): if True, show each figure
    """
    # Clean directory
    if isdir(save_path):
        shutil.rmtree(save_path)
        mkdir(save_path)

    for data, is_new_epoch in dataset:

        # Create feed dictionary for next mini batch
        inputs, _, _, inputs_seq_len, input_names = data
        feed_dict = {
            model.inputs_pl_list[0]: inputs,
            model.inputs_seq_len_pl_list[0]: inputs_seq_len,
            model.keep_prob_pl_list[0]: 1.0
        }

        batch_size, max_frame_num = inputs.shape[:2]
        probs_char = session.run(posteriors_op_main, feed_dict=feed_dict)
        probs_phone = session.run(posteriors_op_sub, feed_dict=feed_dict)
        probs_char = probs_char.reshape(-1, max_frame_num, model.num_classes)
        probs_phone = probs_phone.reshape(-1, max_frame_num, model.num_classes)

        # Visualize
        for i_batch in range(batch_size):
            prob_char = probs_char[i_batch][:int(inputs_seq_len[i_batch]), :]
            prob_phone = probs_phone[i_batch][:int(inputs_seq_len[i_batch]), :]

            plt.clf()
            plt.figure(figsize=(10, 4))
            frame_num = int(inputs_seq_len[i_batch])
            times_probs = np.arange(frame_num) * num_stack / 100

            # NOTE: Blank class is set to the last class in TensorFlow
            # Plot characters
            plt.subplot(211)
            for i in range(0, prob_char.shape[-1] - 1, 1):
                plt.plot(times_probs, prob_char[:, i])
            plt.plot(
                times_probs, prob_char[:, -1], ':', label='blank', color='grey')
            plt.xlabel('Time [sec]', fontsize=12)
            plt.ylabel('Characters', fontsize=12)
            plt.xlim([0, frame_num * num_stack / 100])
            plt.ylim([0.05, 1.05])
            plt.xticks(list(range(0, int(frame_num * num_stack / 100) + 1, 1)))
            plt.yticks(list(range(0, 2, 1)))
            plt.legend(loc="upper right", fontsize=12)

            # Plot phones
            plt.subplot(212)
            for i in range(0, prob_phone.shape[-1] - 1, 1):
                plt.plot(times_probs, prob_phone[:, i])
            plt.plot(
                times_probs, prob_phone[:, -1], ':', label='blank', color='grey')
            plt.xlabel('Time [sec]', fontsize=12)
            plt.ylabel('Phones', fontsize=12)
            plt.xlim([0, frame_num])
            plt.ylim([0.05, 1.05])
            plt.xticks(list(range(0, int(frame_num * num_stack / 100) + 1, 1)))
            plt.yticks(list(range(0, 2, 1)))
            plt.legend(loc="upper right", fontsize=12)

            if show:
                plt.show()

            # Save as a png file
            if save_path is not None:
                plt.savefig(
                    join(save_path, input_names[i_batch] + '.png'), dvi=500)

        if is_new_epoch:
            break


def main():

    args = parser.parse_args()

    # Load config file
    with open(os.path.join(args.model_path, 'config.yml'), "r") as f:
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
    model = MultitaskCTC(
        encoder_type=params['encoder_type'],
        input_size=params['input_size'] * params['num_stack'],
        num_units=params['num_units'],
        num_layers_main=params['num_layers_main'],
        num_layers_sub=params['num_layers_sub'],
        num_classes_main=params['num_classes_main'],
        num_classes_sub=params['num_classes_sub'],
        main_task_weight=params['main_task_weight'],
        lstm_impl=params['lstm_impl'],
        use_peephole=params['use_peephole'],
        parameter_init=params['weight_init'],
        clip_grad_norm=params['clip_grad_norm'],
        clip_activation=params['clip_activation'],
        num_proj=params['num_proj'],
        weight_decay=params['weight_decay'])

    model.save_path = args.model_path
    do_plot(model=model, params=params, epoch=args.epoch)


if __name__ == '__main__':
    main()
