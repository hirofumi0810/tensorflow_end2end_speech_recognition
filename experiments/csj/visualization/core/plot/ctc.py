#! /usr/bin/env python
# -*- coding: utf-8 -*-

""""Utilities for plotting of the CTC model (CSJ corpus)."""

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
        label_type: string, kanji or kana or phone
        save_path: path to save ctc outputs
        show: if True, show each figure
    """
    # Batch size is expected to be 1
    iteration = dataset.data_num

    # Make data generator
    mini_batch = dataset.next_batch(batch_size=1)

    save_path = mkdir_join(save_path, 'ctc_output')

    for step in range(iteration):
        # Create feed dictionary for next mini batch
        inputs, _, inputs_seq_len, input_names = mini_batch.__next__()

        feed_dict = {
            network.inputs: inputs,
            network.inputs_seq_len: inputs_seq_len,
            network.keep_prob_input: 1.0,
            network.keep_prob_hidden: 1.0,
            network.keep_prob_output: 1.0
        }

        # Visualize
        max_frame_num = inputs.shape[1]
        posteriors = session.run(posteriors_op, feed_dict=feed_dict)

        i_batch = 0  # index in mini-batch
        posteriors_index = np.array(
            [i_batch * max_frame_num + i for i in range(max_frame_num)])

        plot_probs_ctc(
            probs=posteriors[posteriors_index][:int(inputs_seq_len[0]), :],
            wav_index=input_names[0],
            label_type=label_type,
            save_path=save_path,
            show=show)


def plot_probs_ctc(probs, wav_index, label_type, save_path, show):
    """Plot posteriors of phones.
    Args:
        probs:
        wav_index: string
        label_type: string, kanji or kana or phone
        save_path: path to save ctc outpus
        show: if True, show each figure
    """
    duration = probs.shape[0]
    times_probs = np.arange(len(probs))
    plt.clf()
    plt.figure(figsize=(10, 4))

    # Blank class is set to the last class in TensorFlow
    if label_type == 'kanji':
        blank_index = 3386
    elif label_type == 'kana':
        blank_index = 147
    elif label_type == 'phone':
        blank_index = 38

    # NOTE:
    # 0: silence(_)
    # 1: noise(NZ)

    # Plot
    plt.plot(times_probs, probs[:, 0],
             label='silence', color='black', linewidth=2)
    for i in range(1, blank_index, 1):
        plt.plot(times_probs, probs[:, i], linewidth=2)
    plt.plot(times_probs, probs[:, blank_index], ':',
             label='blank', color='grey', linewidth=2)
    plt.xlabel('Time[sec]', fontsize=12)
    plt.ylabel(label_type, fontsize=12)
    plt.xlim([0, duration])
    plt.ylim([0.05, 1.05])
    plt.xticks(list(range(0, int(len(probs) / 100) + 1, 1)))
    plt.yticks(list(range(0, 2, 1)))
    plt.legend(loc="upper right", fontsize=12)

    if show:
        plt.show()

    # Save as a png file
    if save_path is not None:
        save_path = join(save_path, wav_index + '.png')
        plt.savefig(save_path, dvi=500)
