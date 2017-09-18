#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilities for plotting of the Attetnion-based model (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
# import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from experiments.utils.data.labels.character import num2char
from experiments.utils.data.labels.phone import num2phone
from experiments.utils.directory import mkdir_join

plt.style.use('ggplot')
sns.set_style("white")

blue = '#4682B4'
orange = '#D2691E'
green = '#006400'


def attention_test(session, decode_op, attention_weights_op, model, dataset,
                   label_type, save_path=None, show=False):
    """Visualize attention weights of Attetnion-based model.
    Args:
        session: session of training model
        decode_op: operation for decoding
        attention_weights_op: operation for computing attention weights
        model: model to evaluate
        dataset: An instance of a `Dataset` class
        label_type (string, optional): phone39 or phone48 or phone61 or character or
            character_capital_divide
        save_path (string, optional): path to save attention weights plotting
        show (bool, optional): if True, show each figure
    """
    save_path = mkdir_join(save_path, 'attention_weights')

    map_file_path = '../metrics/mapping_files/attention/' + label_type + '.txt'

    # Load mapping file
    map_dict = {}
    with open(map_file_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            map_dict[int(line[1])] = line[0]

    while True:

        # Create feed dictionary for next mini batch
        data, is_new_epoch = dataset.next(batch_size=1)
        inputs, _, inputs_seq_len, _, input_names = data
        # NOTE: Batch size is expected to be 1

        feed_dict = {
            model.inputs_pl_list[0]: inputs,
            model.inputs_seq_len_pl_list[0]: inputs_seq_len,
            model.keep_prob_input_pl_list[0]: 1.0,
            model.keep_prob_hidden_pl_list[0]: 1.0,
            model.keep_prob_output_pl_list[0]: 1.0
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

        if is_new_epoch:
            break
