#! /usr/bin/env python
# -*- coding: utf-8 -*-

""""Utilities for plotting of the CTC model (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from experiments.utils.directory import mkdir_join

plt.style.use('ggplot')
sns.set_style("white")

blue = '#4682B4'
orange = '#D2691E'
green = '#006400'


def posterior_test(session, posteriors_op, network, dataset, label_type,
                   save_path=None, show=False):
    """Visualize label posteriors of CTC model.
    Args:
        session: session of training model
        posteriois_op: operation for computing posteriors
        network: network to evaluate
        dataset: An instance of a `Dataset` class
        label_type: string, phone39 or phone48 or phone61 or character or
            character_capital_divide
        save_path: path to save ctc outputs
        show: if True, show each figure
    """
    save_path = mkdir_join(save_path, 'ctc_output')

    # Batch size is expected to be 1
    for data, next_epoch_flag in dataset(batch_size=1):
        # Create feed dictionary for next mini batch
        inputs, _, inputs_seq_len, input_names = data

        feed_dict = {
            network.inputs_pl_list[0]: inputs,
            network.inputs_seq_len_pl_list[0]: inputs_seq_len,
            network.keep_prob_input_pl_list[0]: 1.0,
            network.keep_prob_hidden_pl_list[0]: 1.0,
            network.keep_prob_output_pl_list[0]: 1.0
        }

        # Visualize
        max_frame_num = inputs.shape[1]
        posteriors = session.run(posteriors_op, feed_dict=feed_dict)

        i_batch = 0  # index in mini-batch
        posteriors_index = np.array(
            [i_batch * max_frame_num + i for i in range(max_frame_num)])
        posteriors = posteriors[posteriors_index][:int(inputs_seq_len[0]), :]

        plt.clf()
        plt.figure(figsize=(10, 4))
        duration = posteriors.shape[0]
        times_probs = np.arange(len(posteriors))

        # NOTE: Blank class is set to the last class in TensorFlow
        for i in range(0, posteriors.shape[1] - 1, 1):
            plt.plot(times_probs, posteriors[:, i])
        plt.plot(times_probs, posteriors[:, -1],
                 ':', label='blank', color='grey')
        plt.xlabel('Time [sec]', fontsize=12)
        plt.ylabel('Posteriors', fontsize=12)
        plt.xlim([0, duration])
        plt.ylim([0.05, 1.05])
        plt.xticks(list(range(0, int(len(posteriors) / 100) + 1, 1)))
        plt.yticks(list(range(0, 2, 1)))
        plt.legend(loc="upper right", fontsize=12)

        if show:
            plt.show()

        # Save as a png file
        if save_path is not None:
            plt.savefig(join(save_path, input_names[0] + '.png'), dvi=500)

        if next_epoch_flag:
            break


def posterior_test_multitask(session, posteriors_op_main, posteriors_op_sub,
                             network, dataset, label_type_main, label_type_sub,
                             save_path=None, show=False):
    """Visualize label posteriors of the multi-task CTC model.
    Args:
        session: session of training model
        posteriois_op_main: operation for computing posteriors in the main task
        posteriois_op_sub: operation for computing posteriors in the sub task
        network: network to evaluate
        dataset: An instance of a `Dataset` class
        label_type_sub: string, phone39 or phone48 or phone61
        save_path: path to save ctc outpus
        show: if True, show each figure
    """
    save_path = mkdir_join(save_path, 'ctc_output')

    # Batch size is expected to be 1
    for data, next_epoch_flag in dataset(batch_size=1):
        # Create feed dictionary for next mini batch
        inputs, _, _, inputs_seq_len, input_names = data

        feed_dict = {
            network.inputs_pl_list[0]: inputs,
            network.inputs_seq_len_pl_list[0]: inputs_seq_len,
            network.keep_prob_input_pl_list[0]: 1.0,
            network.keep_prob_hidden_pl_list[0]: 1.0,
            network.keep_prob_output_pl_list[0]: 1.0
        }

        # Visualize
        max_frame_num = inputs.shape[1]
        posteriors_char = session.run(
            posteriors_op_main, feed_dict=feed_dict)
        posteriors_phone = session.run(
            posteriors_op_sub, feed_dict=feed_dict)

        i_batch = 0  # index in mini-batch
        posteriors_index = np.array(
            [i_batch * max_frame_num + i for i in range(max_frame_num)])
        posteriors_char = posteriors_char[posteriors_index][:int(
            inputs_seq_len[0]), :]
        posteriors_phone = posteriors_phone[posteriors_index][:int(
            inputs_seq_len[0]), :]

        plt.clf()
        plt.figure(figsize=(10, 4))
        duration = posteriors_char.shape[0]
        times_probs = np.arange(len(posteriors_char))

        # NOTE: Blank class is set to the last class in TensorFlow
        # Plot characters
        plt.subplot(211)
        for i in range(0, posteriors_char.shape[1] - 1, 1):
            plt.plot(times_probs, posteriors_char[:, i])
        plt.plot(times_probs, posteriors_char[:, -1],
                 ':', label='blank', color='grey')
        plt.xlabel('Time [sec]', fontsize=12)
        plt.ylabel('Characters', fontsize=12)
        plt.xlim([0, duration])
        plt.ylim([0.05, 1.05])
        plt.xticks(list(range(0, int(len(posteriors_char) / 100) + 1, 1)))
        plt.yticks(list(range(0, 2, 1)))
        plt.legend(loc="upper right", fontsize=12)

        # Plot phones
        plt.subplot(212)
        for i in range(0, posteriors_phone.shape[1] - 1, 1):
            plt.plot(times_probs, posteriors_phone[:, i])
        plt.plot(times_probs, posteriors_phone[:, -1],
                 ':', label='blank', color='grey')
        plt.xlabel('Time [sec]', fontsize=12)
        plt.ylabel('Phones', fontsize=12)
        plt.xlim([0, duration])
        plt.ylim([0.05, 1.05])
        plt.xticks(list(range(0, int(len(posteriors_phone) / 100) + 1, 1)))
        plt.yticks(list(range(0, 2, 1)))
        plt.legend(loc="upper right", fontsize=12)

        if show:
            plt.show()

        # Save as a png file
        if save_path is not None:
            plt.savefig(join(save_path, input_names[0] + '.png'), dvi=500)

        if next_epoch_flag:
            break
