#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Save the trained CTC posteriors (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import tensorflow as tf
import yaml
import argparse

sys.path.append(os.path.abspath('../../../'))
from experiments.librispeech.data.load_dataset_ctc import Dataset
from models.ctc.vanilla_ctc import CTC
from utils.directory import mkdir_join

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--model_path', type=str,
                    help='path to the model to evaluate')
parser.add_argument('--eval_batch_size', type=str, default=1,
                    help='the size of mini-batch in evaluation')


def do_save(model, params, epoch, eval_batch_size):
    """Save the CTC outputs.
    Args:
        model: the model to restore
        params (dict): A dictionary of parameters
        epoch (int): the epoch to restore
        eval_batch_size (int): the size of mini-batch in evaluation
    """
    # Load dataset
    train_data = Dataset(
        data_type='train', train_data_size=params['train_data_size'],
        label_type=params['label_type'],
        batch_size=eval_batch_size,
        splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=True)

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
        posteriors_op = model.posteriors(logits, blank_prior=1)

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

        for data, is_new_epoch in train_data:

            # Create feed dictionary for next mini batch
            inputs, _, inputs_seq_len, input_names = data
            feed_dict = {
                model.inputs_pl_list[0]: inputs[0],
                model.inputs_seq_len_pl_list[0]: inputs_seq_len[0],
                model.keep_prob_input_pl_list[0]: 1.0,
                model.keep_prob_hidden_pl_list[0]: 1.0,
                model.keep_prob_output_pl_list[0]: 1.0
            }

            batch_size, max_frame_num = inputs[0].shape[:2]
            posteriors = sess.run(posteriors_op, feed_dict=feed_dict)
            posteriors = posteriors.reshape(-1,
                                            max_frame_num, model.num_classes)

            for i_batch in range(batch_size):
                prob = posteriors[i_batch][:int(inputs_seq_len[0][i_batch]), :]

                # Save as a npy file
                np.save(mkdir_join(model.save_path, 'probs',
                                   input_names[0][i_batch]), prob)

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
    do_save(model=model, params=params, epoch=args.epoch)


if __name__ == '__main__':
    main()
