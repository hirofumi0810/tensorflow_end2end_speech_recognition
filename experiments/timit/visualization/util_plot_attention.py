#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilities for plotting of Attetnion-based model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from utils.labels.character import num2char
from utils.labels.phone import num2phone
from utils.directory import mkdir_join

plt.style.use('ggplot')
sns.set_style("white")

blue = '#4682B4'
orange = '#D2691E'
green = '#006400'


def attention_test(session, decode_op, attention_weights_op, network, dataset,
                   label_type, save_path=None, show=False):
    """Visualize attention weights of Attetnion-based model.
    Args:
        session: session of training model
        decode_op: operation for decoding
        attention_weights_op: operation for computing attention weights
        network: network to evaluate
        dataset: An instance of a `Dataset` class
        label_type: string, phone39 or phone48 or phone61 or character
        save_path: path to save attention weights plotting
        show: if True, show each figure
    """
    # Batch size is expected to be 1
    iteration = dataset.data_num

    # Make data generator
    mini_batch = dataset.next_batch(batch_size=1)

    save_path = mkdir_join(save_path, 'attention_weights')

    if label_type == 'character':
        map_file_path = '../metric/mapping_files/attention/char2num.txt'
    else:
        map_file_path = '../metric/mapping_files/attention/phone2num_' + \
            label_type[5:7] + '.txt'

    # Load mapping file
    map_dict = {}
    with open(map_file_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            map_dict[int(line[1])] = line[0]

    for step in range(iteration):
        # Create feed dictionary for next mini batch
        inputs, _, inputs_seq_len, _, input_names = mini_batch.__next__()

        feed_dict = {
            network.inputs: inputs,
            network.inputs_seq_len: inputs_seq_len,
            network.keep_prob_input: 1.0,
            network.keep_prob_hidden: 1.0
        }

        # Visualize
        attention_weights_list, predicted_ids = session.run(
            [attention_weights_op, decode_op], feed_dict=feed_dict)

        attention_weights = attention_weights_list[0]
        labels_seq_len, inputs_seq_len = attention_weights.shape

        # Check if the sum of attention weights equals to 1
        # print(np.sum(attention_weights, axis=1))

        # Convert from indices to the corresponding labels
        label_list = []
        for i_output in range(len(predicted_ids[0])):
            label_list.append(map_dict[predicted_ids[0, i_output]])

        plt.clf()
        plt.figure(figsize=(10, 4))
        sns.heatmap(attention_weights,
                    cmap='Blues',
                    xticklabels=False,
                    yticklabels=label_list)

        plt.xlabel('Input frames', fontsize=12)
        plt.ylabel('Output labels (top to bottom)', fontsize=12)
        if show:
            plt.show()

        # Save as a png file
        if save_path is not None:
            save_path = join(save_path, input_names[0] + '.png')
            plt.savefig(save_path, dvi=500)
