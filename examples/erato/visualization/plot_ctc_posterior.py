#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot the trained CTC posteriors (ERATO corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath, isdir
import sys
import tensorflow as tf
import yaml
import argparse
import shutil

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
from experiments.erato.data.load_dataset_ctc import Dataset
from models.ctc.ctc import CTC
from utils.directory import mkdir_join, mkdir

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--model_path', type=str,
                    help='path to the model to evaluate')
parser.add_argument('--beam_width', type=int, default=20,
                    help='beam_width (int, optional): beam width for beam search.' +
                    ' 1 disables beam search, which mean greedy decoding.')
parser.add_argument('--eval_batch_size', type=str, default=1,
                    help='the size of mini-batch in evaluation')


def do_plot(model, params, epoch, beam_width, eval_batch_size):
    """Decode the CTC outputs.
    Args:
        model: the model to restore
        params (dict): A dictionary of parameters
        epoch (int): the epoch to restore
        beam_width (int): beam width for beam search.
            1 disables beam search, which mean greedy decoding.
        eval_batch_size (int): the size of mini-batch when evaluation
    """
    # Load dataset
    test_data = Dataset(
        data_type='test', label_type=params['label_type'],
        ss_type=params['ss_type'],
        batch_size=eval_batch_size, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=False, progressbar=True)

    # Define placeholders
    model.create_placeholders()

    # Add to the graph each operation (including model definition)
    _, logits = model.compute_loss(model.inputs_pl_list[0],
                                   model.labels_pl_list[0],
                                   model.inputs_seq_len_pl_list[0],
                                   model.keep_prob_pl_list[0])
    posteriors_op = model.posteriors(logits)

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
             posteriors_op=posteriors_op,
             model=model,
             dataset=test_data,
             label_type=params['label_type'],
             ss_type=params['ss_type'],
             num_stack=params['num_stack'],
             #  save_path=mkdir_join(model.save_path, 'ctc_output'))
             save_path=None)


def plot(session, posteriors_op, model, dataset, label_type, ss_type,
         num_stack=1, save_path=None, show=False):
    """Plot posteriors of phones.
    Args:
        session: session of training model
        posteriois_op: operation for computing posteriors
        model: the model to evaluate
        dataset: An instance of a `Dataset` class
        label_type (string): kana
        ss_type (string):
        num_stack (int): the number of frames to stack
        save_path (string, string): path to save ctc outputs
        show (bool, optional): if True, show each figure
    """
    # Blank class is set to the last class in TensorFlow
    if label_type == 'kana':
        if ss_type == 'remove':
            blank_index = 147
        elif ss_type == 'insert_both':
            blank_index = 155
        else:
            blank_index = 151
        laughter_index = 147
        filler_index = 148
        backchannel_index = 149
        disfluency_index = 150

    # Clean directory
    if isdir(save_path):
        shutil.rmtree(save_path)
        mkdir(save_path)

    for data, is_new_epoch in dataset:

        # Create feed dictionary for next mini batch
        inputs, _, inputs_seq_len, input_names = data

        feed_dict = {
            model.inputs_pl_list[0]: inputs,
            model.inputs_seq_len_pl_list[0]: inputs_seq_len,
            model.keep_prob_pl_list[0]: 1.0
        }

        # Visualize
        batch_size, max_frame_num = inputs.shape[:2]
        probs = session.run(posteriors_op, feed_dict=feed_dict)
        probs = probs.reshape(-1, max_frame_num, model.num_classes)

        # Visualize
        for i_batch in range(batch_size):
            prob = probs[i_batch][:int(inputs_seq_len[0]), :]

            plt.clf()
            plt.figure(figsize=(10, 4))
            frame_num = int(inputs_seq_len[i_batch])
            times_probs = np.arange(frame_num) * num_stack / 100

            # NOTE: Blank class is set to the last class in TensorFlow
            for i in range(0, prob.shape[-1] - 1, 1):
                plt.plot(times_probs, prob[:, i])
            plt.plot(times_probs, prob[:, -1],
                     ':', label='blank', color='grey')
            plt.xlabel('Time [sec]', fontsize=12)
            plt.ylabel('Posteriors', fontsize=12)
            plt.xlim([0, frame_num * num_stack / 100])
            plt.ylim([0.05, 1.05])
            plt.xticks(list(range(0, int(frame_num * num_stack / 100) + 1, 1)))
            plt.yticks(list(range(0, 2, 1)))
            plt.legend(loc="upper right", fontsize=12)

            if show:
                plt.show()

            # Save as a png file
            if save_path is not None:
                plt.savefig(join(save_path, input_names[0] + '.png'), dvi=500)

        if is_new_epoch:
            break

        # # Plot
        # if ss_type != 'remove':
        #     plt.subplot(211)
        #     plt.plot(times_probs, probs[:, laughter_index],
        #              label='Laughter', color=orange, linewidth=2)
        #     plt.plot(times_probs, probs[:, filler_index],
        #              label='Filler', color=green, linewidth=2)
        #     plt.plot(times_probs, probs[:, backchannel_index],
        #              label='Backchannel', color=blue, linewidth=2)
        #     plt.plot(times_probs, probs[:, disfluency_index],
        #              label='Disfluency', color='magenta', linewidth=2)
        #     plt.xlabel('Time[sec]', fontsize=12)
        #     plt.ylabel('social signals', fontsize=12)
        #     plt.xlim([0, duration])
        #     plt.ylim([0.05, 1.05])
        #     plt.xticks(list(range(0, int(len(probs) / 100) + 1, 1)))
        #     plt.yticks(list(range(0, 2, 1)))
        #     plt.legend(loc="upper right", fontsize=12)
        #
        #     # not social signals
        #     plt.subplot(212)
        #
        #     plt.plot(times_probs, probs[:, 0],
        #              label='Silence', color='black', linewidth=2)
        #     for i in range(1, blank_index, 1):
        #         plt.plot(times_probs, probs[:, i], linewidth=2)
        #     plt.plot(times_probs, probs[:, blank_index], ':',
        #              label='Blank', color='grey', linewidth=2)
        #     plt.xlabel('Time[sec]', fontsize=12)
        #     plt.ylabel(label_type, fontsize=12)
        #     plt.xlim([0, duration])
        #     plt.ylim([0.05, 1.05])
        #     plt.xticks(list(range(0, int(len(probs) / 100) + 1, 1)))
        #     plt.yticks(list(range(0, 2, 1)))
        #     plt.legend(loc="upper right", fontsize=12)
        #
        #     if show:
        #         plt.show()
        #
        #     # Save as a png file
        #     if save_path is not None:
        #         save_path = join(save_path, wav_index + '.png')
        #         plt.savefig(save_path, dvi=500)


def main():

    args = parser.parse_args()

    # Load config file
    with open(join(args.model_path, 'config.yml'), "r") as f:
        config = yaml.load(f)
        params = config['param']

    # Except for a blank label
    if params['ss_type'] == 'remove':
        params['num_classes'] = 147
    elif params['ss_type'] == 'insert_left':
        params['num_classes'] = 151
    elif params['ss_type'] == 'insert_both':
        params['num_classes'] = 155

    # Model setting
    model = CTC(encoder_type=params['encoder_type'],
                input_size=params['input_size'],
                splice=params['splice'],
                num_stack=params['num_stack'],
                num_units=params['num_units'],
                num_layers=params['num_layers'],
                num_classes=params['num_classes'],
                lstm_impl=params['lstm_impl'],
                use_peephole=params['use_peephole'],
                parameter_init=params['weight_init'],
                clip_grad_norm=params['clip_grad_norm'],
                clip_activation=params['clip_activation'],
                num_proj=params['num_proj'],
                weight_decay=params['weight_decay'])

    model.save_path = args.model_path
    do_plot(model=model, params=params,
            epoch=args.epoch, beam_width=args.beam_width,
            eval_batch_size=args.eval_batch_size)


if __name__ == '__main__':
    main()
