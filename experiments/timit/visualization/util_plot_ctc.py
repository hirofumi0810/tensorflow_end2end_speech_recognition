#! /usr/bin/env python
# -*- coding: utf-8 -*-

""""Utilities for plotting of CTC model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils.directory import mkdir_join

plt.style.use('ggplot')
sns.set_style("white")

blue = '#4682B4'
orange = '#D2691E'
green = '#006400'


def posterior_test(session, posteriors_op, network, dataset, label_type,
                   save_path=None):
    """Visualize label posteriors of CTC model.
    Args:
        session: session of training model
        posteriois_op: operation for computing posteriors
        network: network to evaluate
        dataset: Dataset class
        label_type: phone39 or phone48 or phone61 or character
        save_path: path to save ctc outputs
    """
    save_path = mkdir_join(save_path, 'ctc_output')
    batch_size = 1
    num_examples = dataset.data_num
    iteration = int(num_examples / batch_size)
    if (num_examples / batch_size) != int(num_examples / batch_size):
        iteration += 1

    for step in range(iteration):
        # Create feed dictionary for next mini batch
        inputs, _, seq_len, input_names = dataset.next_batch(
            batch_size=batch_size)

        feed_dict = {
            network.inputs: inputs,
            network.seq_len: seq_len,
            network.keep_prob_input: 1.0,
            network.keep_prob_hidden: 1.0
        }

        # Visualize
        batch_size_each = len(seq_len)
        max_frame_num = inputs.shape[1]
        posteriors = session.run(posteriors_op, feed_dict=feed_dict)
        for i_batch in range(batch_size_each):
            posteriors_index = np.array([i_batch + (batch_size_each * j)
                                         for j in range(max_frame_num)])
            if label_type != 'character':
                plot_probs_ctc_phone(
                    probs=posteriors[posteriors_index][:int(
                        seq_len[i_batch]), :],
                    wav_index=input_names[i_batch],
                    label_type=label_type,
                    save_path=save_path)
            else:
                plot_probs_ctc_char(
                    probs=posteriors[posteriors_index][:int(
                        seq_len[i_batch]), :],
                    wav_index=input_names[i_batch],
                    save_path=save_path)


def posterior_test_multitask(session, posteriors_op_main, posteriors_op_second, network,
                             dataset, label_type_second, save_path=None):
    """Visualize label posteriors of Multi-task CTC model.
    Args:
        session: session of training model
        posteriois_op_main: operation for computing posteriors in the main task
        posteriois_op_second: operation for computing posteriors in the second
            task
        network: network to evaluate
        dataset: Dataset class
        label_type_second: phone39 or phone48 or phone61
        save_path: path to save ctc outpus
    """
    save_path = mkdir_join(save_path, 'ctc_output')
    batch_size = 1
    num_examples = dataset.data_num
    iteration = int(num_examples / batch_size)
    if (num_examples / batch_size) != int(num_examples / batch_size):
        iteration += 1

    for step in range(iteration):
        # Create feed dictionary for next mini batch
        inputs, _, _, seq_len, input_names = dataset.next_batch(
            batch_size=batch_size)

        feed_dict = {
            network.inputs: inputs,
            network.seq_len: seq_len,
            network.keep_prob_input: 1.0,
            network.keep_prob_hidden: 1.0
        }

        # Visualize
        batch_size_each = len(seq_len)
        max_frame_num = inputs.shape[1]
        posteriors_char = session.run(
            posteriors_op_main, feed_dict=feed_dict)
        posteriors_phone = session.run(
            posteriors_op_second, feed_dict=feed_dict)
        for i_batch in range(batch_size_each):
            posteriors_index = np.array([i_batch + (batch_size_each * j)
                                         for j in range(max_frame_num)])

            plot_probs_ctc_char_phone(
                probs_char=posteriors_char[posteriors_index][:int(
                    seq_len[i_batch]), :],
                probs_phone=posteriors_phone[posteriors_index][:int(
                    seq_len[i_batch]), :],
                wav_index=input_names[i_batch],
                label_type_second=label_type_second,
                save_path=save_path)


def plot_probs_ctc_phone(probs, wav_index, label_type, save_path=None):
    """Plot posteriors of phones.
    Args:
        probs:
        wav_index: int
        label_type: phone39 or phone48 or phone61
        save_path:
    """
    duration = probs.shape[0]
    times_probs = np.arange(len(probs))
    plt.clf()
    plt.figure(figsize=(10, 4))

    # Blank class is set to the last class in TensorFlow
    if label_type == 'phone39':
        blank_index = 39
    elif label_type == 'phone48':
        blank_index = 48
    elif label_type == 'phone61':
        blank_index = 61

    # Plot phones
    plt.plot(times_probs, probs[:, 0],
             label='silence', color='black', linewidth=2)
    for i in range(1, blank_index, 1):
        plt.plot(times_probs, probs[:, i])
    plt.plot(times_probs, probs[:, blank_index],
             ':', label='blank', color='grey')
    plt.xlabel('Time[sec]', fontsize=12)
    plt.ylabel('Phones', fontsize=12)
    plt.xlim([0, duration])
    plt.ylim([0.05, 1.05])
    plt.xticks(list(range(0, int(len(probs) / 100) + 1, 1)))
    plt.yticks(list(range(0, 2, 1)))
    plt.legend(loc="upper right", fontsize=12)
    # plt.show()

    # Save as a png file
    if save_path is not None:
        save_path = join(save_path, wav_index + '.png')
        plt.savefig(save_path, dvi=500)


def plot_probs_ctc_char(probs, wav_index, save_path=None):
    """Plot posteriors of characters.
    Args:
        probs:
        wav_index: int
        save_path:
    """
    duration = probs.shape[0]
    times_probs = np.arange(len(probs))
    plt.clf()
    plt.figure(figsize=(10, 4))

    # Blank class is set to the last class in TensorFlow
    blank_index = 30

    # Plot characters
    plt.plot(times_probs, probs[:, 0],
             label='silence', color='black', linewidth=2)
    for i in range(1, blank_index, 1):
        plt.plot(times_probs, probs[:, i])
    plt.plot(times_probs, probs[:, blank_index],
             ':', label='blank', color='grey')
    plt.xlabel('Time[sec]', fontsize=12)
    plt.ylabel('Characters', fontsize=12)
    plt.xlim([0, duration])
    plt.ylim([0.05, 1.05])
    plt.xticks(list(range(0, int(len(probs) / 100) + 1, 1)))
    plt.yticks(list(range(0, 2, 1)))
    plt.legend(loc="upper right", fontsize=12)
    # plt.show()

    # Save as a png file
    if save_path is not None:
        save_path = join(save_path, wav_index + '.png')
        plt.savefig(save_path, dvi=500)


def plot_probs_ctc_char_phone(probs_char, probs_phone, wav_index,
                              label_type_second, save_path=None):
    """Plot posteriors of characters and phones.
    Args:
        probs_char:
        probs_phone:
        wav_index: int
        label_type_second:
        save_path:
    """
    duration = probs_char.shape[0]
    times_probs = np.arange(len(probs_char))
    plt.clf()
    plt.figure(figsize=(10, 4))

    # Blank class is set to the last class in TensorFlow
    blank_index_char = 30
    if label_type_second == 'phone39':
        blank_index_phone = 39
    elif label_type_second == 'phone48':
        blank_index_phone = 48
    elif label_type_second == 'phone61':
        blank_index_phone = 61

    # Plot characters
    plt.subplot(211)
    plt.plot(times_probs, probs_char[:, 0],
             label='silence', color='black', linewidth=2)
    for i in range(1, blank_index_char, 1):
        plt.plot(times_probs, probs_char[:, i])
    plt.plot(times_probs, probs_char[:, blank_index_char],
             ':', label='blank', color='grey')
    plt.xlabel('Time[sec]', fontsize=12)
    plt.ylabel('Characters', fontsize=12)
    plt.xlim([0, duration])
    plt.ylim([0.05, 1.05])
    plt.xticks(list(range(0, int(len(probs_char) / 100) + 1, 1)))
    plt.yticks(list(range(0, 2, 1)))
    plt.legend(loc="upper right", fontsize=12)

    # Plot phones
    plt.subplot(212)
    plt.plot(times_probs, probs_phone[:, 0],
             label='silence', color='black', linewidth=2)
    for i in range(1, blank_index_phone, 1):
        plt.plot(times_probs, probs_phone[:, i])
    plt.plot(times_probs, probs_phone[:, blank_index_phone],
             ':', label='blank', color='grey')
    plt.xlabel('Time[sec]', fontsize=12)
    plt.ylabel('Phones', fontsize=12)
    plt.xlim([0, duration])
    plt.ylim([0.05, 1.05])
    plt.xticks(list(range(0, int(len(probs_phone) / 100) + 1, 1)))
    plt.yticks(list(range(0, 2, 1)))
    plt.legend(loc="upper right", fontsize=12)
    # plt.show()

    # Save as a png file
    if save_path is not None:
        save_path = join(save_path, wav_index + '.png')
        plt.savefig(save_path, dvi=500)
