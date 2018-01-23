#! /usr/bin/env python
# -*- coding: utf-8 -*-

""""Utilities for plotting of the CTC model (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
sns.set_style("white")

from utils.directory import mkdir_join


blue = '#4682B4'
orange = '#D2691E'
green = '#006400'


def posterior_test_multitask(session, posteriors_op_main, posteriors_op_sub,
                             model, dataset, label_type_main, label_type_sub,
                             save_path=None, show=False):
    """Visualize label posteriors of the multi-task CTC model.
    Args:
        session: session of training model
        posteriois_op_main: operation for computing posteriors in the main task
        posteriois_op_sub: operation for computing posteriors in the sub task
        model: model to evaluate
        dataset: An instance of a `Dataset` class
        label_type_sub (string): phone39 or phone48 or phone61
        save_path (string): path to save ctc outpus
        show (bool, optional): if True, show each figure
    """
    save_path = mkdir_join(save_path, 'ctc_output')

    # NOTE: Batch size is expected to be 1
    while True:

        # Create feed dictionary for next mini batch
        data, is_new_epoch = dataset.next(batch_size=1)
        inputs, _, _, inputs_seq_len, input_names = data

        feed_dict = {
            model.inputs_pl_list[0]: inputs[0],
            model.inputs_seq_len_pl_list[0]: inputs_seq_len[0],
            model.keep_prob_input_pl_list[0]: 1.0,
            model.keep_prob_hidden_pl_list[0]: 1.0,
            model.keep_prob_output_pl_list[0]: 1.0
        }

        # Visualize
        max_frame_num = inputs[0].shape[1]
        posteriors_char = session.run(posteriors_op_main, feed_dict=feed_dict)
        posteriors_phone = session.run(posteriors_op_sub, feed_dict=feed_dict)

        i_batch = 0  # index in mini-batch
        posteriors_index = np.array(
            [i_batch * max_frame_num + i for i in range(max_frame_num)])
        posteriors_char = posteriors_char[posteriors_index][:int(
            inputs_seq_len[0][0]), :]
        posteriors_phone = posteriors_phone[posteriors_index][:int(
            inputs_seq_len[0][0]), :]

        plt.clf()
        plt.figure(figsize=(10, 4))
        frame_num = posteriors_char.shape[0]
        times_probs = np.arange(len(posteriors_char))

        # NOTE: Blank class is set to the last class in TensorFlow
        # Plot characters
        plt.subplot(211)
        for i in range(0, posteriors_char.shape[1] - 1, 1):
            plt.plot(times_probs, posteriors_char[:, i])
        plt.plot(
            times_probs, posteriors_char[:, -1], ':', label='blank', color='grey')
        plt.xlabel('Time [sec]', fontsize=12)
        plt.ylabel('Characters', fontsize=12)
        plt.xlim([0, frame_num])
        plt.ylim([0.05, 1.05])
        plt.xticks(list(range(0, int(len(posteriors_char) / 100) + 1, 1)))
        plt.yticks(list(range(0, 2, 1)))
        plt.legend(loc="upper right", fontsize=12)

        # Plot phones
        plt.subplot(212)
        for i in range(0, posteriors_phone.shape[1] - 1, 1):
            plt.plot(times_probs, posteriors_phone[:, i])
        plt.plot(
            times_probs, posteriors_phone[:, -1], ':', label='blank', color='grey')
        plt.xlabel('Time [sec]', fontsize=12)
        plt.ylabel('Phones', fontsize=12)
        plt.xlim([0, frame_num])
        plt.ylim([0.05, 1.05])
        plt.xticks(list(range(0, int(len(posteriors_phone) / 100) + 1, 1)))
        plt.yticks(list(range(0, 2, 1)))
        plt.legend(loc="upper right", fontsize=12)

        if show:
            plt.show()

        # Save as a png file
        if save_path is not None:
            plt.savefig(join(save_path, input_names[0][0] + '.png'), dvi=500)

        if is_new_epoch:
            break
