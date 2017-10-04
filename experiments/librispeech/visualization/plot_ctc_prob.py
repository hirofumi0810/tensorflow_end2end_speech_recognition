#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot the trained CTC posteriors (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
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

sys.path.append(os.path.abspath('../../../'))
from experiments.librispeech.data.load_dataset_ctc import Dataset
from models.ctc.vanilla_ctc import CTC
from utils.directory import mkdir_join

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--model_path', type=str,
                    help='path to the model to evaluate')


def do_decode(model, params, epoch):
    """Decode the CTC outputs.
    Args:
        model: the model to restore
        params (dict): A dictionary of parameters
        epoch (int): the epoch to restore
    """
    # Load dataset
    test_clean_data = Dataset(
        data_type='test_clean',
        train_data_size=params['train_data_size'],
        label_type=params['label_type'],
        batch_size=1, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=False)
    test_other_data = Dataset(
        data_type='test_other',
        train_data_size=params['train_data_size'],
        label_type=params['label_type'],
        batch_size=1, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=False)

    with tf.name_scope('tower_gpu0'):
        # Define placeholders
        model.create_placeholders()

        # Add to the graph each operation (including model definition)
        _, logits = model.compute_loss(
            model.inputs_pl_list[0],
            model.labels_pl_list[0],
            model.inputs_seq_len_pl_list[0],
            model.keep_prob_input_pl_list[0],
            model.keep_prob_hidden_pl_list[0],
            model.keep_prob_output_pl_list[0],
            softmax_temperature=params['softmax_temperature'])
        posteriors_op = model.posteriors(logits,
                                         blank_prior=1,
                                         softmax_temperature=1)

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

        # Visualize
        posterior_test(session=sess,
                       posteriors_op=posteriors_op,
                       model=model,
                       dataset=test_clean_data,
                       label_type=params['label_type'],
                       num_stack=params['num_stack'],
                       #    save_path=None)
                       save_path=mkdir_join(model.save_path, 'ctc_output', 'test-clean'))

        posterior_test(session=sess,
                       posteriors_op=posteriors_op,
                       model=model,
                       dataset=test_other_data,
                       label_type=params['label_type'],
                       num_stack=params['num_stack'],
                       #    save_path=None)
                       save_path=mkdir_join(model.save_path, 'ctc_output', 'test-other'))


def posterior_test(session, posteriors_op, model, dataset, label_type,
                   num_stack=1, save_path=None, show=False):
    """Visualize label posteriors of CTC model.
    Args:
        session: session of training model
        posteriois_op: operation for computing posteriors
        model: the model to evaluate
        dataset: An instance of a `Dataset` class
        label_type (string): phone39 or phone48 or phone61 or character or
            character_capital_divide
        num_stack (int): the number of frames to stack
        save_path (string, string): path to save ctc outputs
        show (bool, optional): if True, show each figure
    """
    while True:

        # Create feed dictionary for next mini batch
        data, is_new_epoch = dataset.next(batch_size=1)
        inputs, _, inputs_seq_len, input_names = data
        # NOTE: Batch size is expected to be 1

        feed_dict = {
            model.inputs_pl_list[0]: inputs[0],
            model.inputs_seq_len_pl_list[0]: inputs_seq_len[0],
            model.keep_prob_input_pl_list[0]: 1.0,
            model.keep_prob_hidden_pl_list[0]: 1.0,
            model.keep_prob_output_pl_list[0]: 1.0
        }

        # Visualize
        max_frame_num = inputs[0].shape[1]
        posteriors = session.run(posteriors_op, feed_dict=feed_dict)

        i_batch = 0  # index in mini-batch
        posteriors_index = np.array(
            [i_batch * max_frame_num + i for i in range(max_frame_num)])
        posteriors = posteriors[posteriors_index][:int(
            inputs_seq_len[0][0]), :]

        plt.clf()
        plt.figure(figsize=(10, 4))
        frame_num = posteriors.shape[0]
        times_probs = np.arange(frame_num) * num_stack / 100

        # NOTE: Blank class is set to the last class in TensorFlow
        for i in range(0, posteriors.shape[1] - 1, 1):
            plt.plot(times_probs, posteriors[:, i])
        plt.plot(times_probs, posteriors[:, -1],
                 ':', label='blank', color='grey')
        plt.xlabel('Time [sec]', fontsize=12)
        plt.ylabel('Posteriors', fontsize=12)
        plt.xlim([0, frame_num * num_stack / 100])
        plt.ylim([0.05, 1.05])
        plt.xticks(list(range(0, int(frame_num * num_stack / 100) + 1, 1)))
        plt.yticks(list(range(0, 2, 1)))
        plt.legend(loc="upper right", fontsize=12)

        plt.show()

        # Save as a png file
        if save_path is not None:
            plt.savefig(os.path.join(
                save_path, input_names[0][0] + '.png'), dvi=500)

        if is_new_epoch:
            break


def main():

    args = parser.parse_args()

    # Load config file
    with open(os.path.join(args.model_path, 'config.yml'), "r") as f:
        config = yaml.load(f)
        params = config['param']

    # Except for a blank class
    if params['label_type'] == 'character':
        params['num_classes'] = 28
    elif params['label_type'] == 'character_capital_divide':
        params['num_classes'] = 77
    elif params['label_type'] == 'word':
        if params['train_data_size'] == 'train_clean100':
            params['num_classes'] = 7213
        elif params['train_data_size'] == 'train_clean360':
            params['num_classes'] = 16287
        elif params['train_data_size'] == 'train_other500':
            params['num_classes'] = 18669
        elif params['train_data_size'] == 'train_all':
            params['num_classes'] = 26642

    # Model setting
    model = CTC(
        encoder_type=params['encoder_type'],
        input_size=params['input_size'] * params['num_stack'],
        num_units=params['num_units'],
        num_layers=params['num_layers'],
        num_classes=params['num_classes'],
        lstm_impl=params['lstm_impl'],
        use_peephole=params['use_peephole'],
        parameter_init=params['weight_init'],
        clip_grad=params['clip_grad'],
        clip_activation=params['clip_activation'],
        num_proj=params['num_proj'],
        weight_decay=params['weight_decay'])

    model.save_path = args.model_path
    do_decode(model=model, params=params, epoch=args.epoch)


if __name__ == '__main__':
    main()
