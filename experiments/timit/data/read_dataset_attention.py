#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Read dataset for Attention-based model (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, basename
import pickle
import random
import numpy as np
import tensorflow as tf

from utils.progressbar import wrap_iterator


class DataSet(object):
    """Read dataset."""

    def __init__(self, data_type, label_type, eos_index, is_sorted=True,
                 is_progressbar=False, num_gpu=1):
        """
        Args:
            data_type: train or dev or test
            label_type: phone39 or phone48 or phone61 or character
            eos_index: int , the index of <EOS> class
            is_sorted: if True, sort dataset by frame num
            is_progressbar: if True, visualize progressbar
        """
        if data_type not in ['train', 'dev', 'test']:
            raise ValueError('data_type is "train" or "dev" or "test".')

        self.data_type = data_type
        self.label_type = label_type
        self.eos_index = eos_index
        self.is_sorted = is_sorted
        self.is_progressbar = is_progressbar

        self.input_size = 123
        self.dataset_path = join(
            '/n/sd8/inaguma/corpus/timit/dataset/attention/',
            label_type, data_type)

        # Load the frame number dictionary
        self.frame_num_dict_path = join(
            self.dataset_path, 'frame_num_dict.pickle')
        with open(self.frame_num_dict_path, 'rb') as f:
            self.frame_num_dict = pickle.load(f)

        # Sort paths to input & label by frame num
        self.frame_num_tuple_sorted = sorted(
            self.frame_num_dict.items(), key=lambda x: x[1])
        input_paths, label_paths = [], []
        for input_name, frame_num in self.frame_num_tuple_sorted:
            input_paths.append(join(
                self.dataset_path, 'input', input_name + '.npy'))
            label_paths.append(join(
                self.dataset_path, 'label', input_name + '.npy'))
        self.input_paths = np.array(input_paths)
        self.label_paths = np.array(label_paths)
        self.data_num = len(self.input_paths)

        # Load all dataset
        print('=> Loading ' + data_type + ' dataset (' + label_type + ')...')
        input_list, label_list = [], []
        for i in wrap_iterator(range(self.data_num), self.is_progressbar):
            input_list.append(np.load(self.input_paths[i]))
            label_list.append(np.load(self.label_paths[i]))
        self.input_list = np.array(input_list)
        self.label_list = np.array(label_list)

        self.rest = set([i for i in range(self.data_num)])

    def next_batch(self, batch_size=None, session=None):
        """Make mini-batch.
        Args:
            batch_size: int, the size of mini-batch
            session:
        Returns:
            inputs: list of input data, size `[batch_size]`
            labels: list of tuple `(indices, values, shape)` of size
                `[batch_size]`
            inputs_seq_len: list of length of inputs of size `[batch_size]`
            labels_seq_len: list of length of target labels of size
                `[batch_size]`
            input_names: list of file name of input data of size `[batch_size]`
        """
        if session is None and self.num_gpu != 1:
            raise ValueError('Set session when using multiple GPUs.')

        if batch_size is None:
            batch_size = self.batch_size

        #########################
        # sorted dataset
        #########################
        if self.is_sorted:
            if len(self.rest) > batch_size:
                sorted_indices = list(self.rest)[:batch_size]
                self.rest -= set(sorted_indices)
            else:
                sorted_indices = list(self.rest)
                self.rest = set([i for i in range(self.data_num)])
                if self.data_type == 'train':
                    print('---Next epoch---')

            # Compute max frame num in mini batch
            max_frame_num = self.input_list[sorted_indices[-1]].shape[0]

            # Compute max target label length in mini batch
            max_seq_len = max(map(len, self.label_list[sorted_indices]))

            # Shuffle selected mini batch (0 ~ len(self.rest)-1)
            random.shuffle(sorted_indices)

            # Initialization
            inputs = np.zeros(
                (len(sorted_indices), max_frame_num, self.input_size))
            labels = np.array([[self.eos_index] * max_seq_len]
                              * len(sorted_indices), dtype=int)
            inputs_seq_len = np.zeros((len(sorted_indices),), dtype=int)
            labels_seq_len = np.zeros((len(sorted_indices),), dtype=int)
            input_names = [None] * len(sorted_indices)

            # Set values of each data in mini batch
            for i_batch, x in enumerate(sorted_indices):
                data_i = self.input_list[x]
                frame_num = data_i.shape[0]
                inputs[i_batch, :frame_num, :] = data_i
                labels[i_batch, :len(self.label_list[x])] = self.label_list[x]
                inputs_seq_len[i_batch] = frame_num
                labels_seq_len[i_batch] = len(self.label_list[x])
                input_names[i_batch] = basename(
                    self.input_paths[x]).split('.')[0]

        #########################
        # not sorted dataset
        #########################
        else:
            if len(self.rest) > batch_size:
                # Randomly sample mini batch
                random_indices = random.sample(list(self.rest), batch_size)
                self.rest -= set(random_indices)
            else:
                random_indices = list(self.rest)
                self.rest = set([i for i in range(self.data_num)])
                if self.data_type == 'train':
                    print('---Next epoch---')

                # Shuffle selected mini batch (0 ~ len(self.rest)-1)
                random.shuffle(random_indices)

            # Compute max frame num in mini batch
            frame_num_list = []
            for data_i in self.input_list[random_indices]:
                frame_num_list.append(data_i.shape[0])
            max_frame_num = max(frame_num_list)

            # Compute max target label length in mini batch
            max_seq_len = max(map(len, self.label_list[random_indices]))

            # Initialization
            inputs = np.zeros(
                (len(random_indices), max_frame_num, self.input_size))
            labels = np.array([[self.eos_index] * max_seq_len]
                              * len(random_indices), dtype=int)
            inputs_seq_len = np.zeros((len(random_indices),), dtype=int)
            labels_seq_len = np.zeros((len(random_indices),), dtype=int)
            input_names = [None] * len(random_indices)

            # Set values of each data in mini batch
            for i_batch, x in enumerate(random_indices):
                data_i = self.input_list[x]
                frame_num = data_i.shape[0]
                inputs[i_batch, :frame_num, :] = data_i
                labels[i_batch, :len(self.label_list[x])] = self.label_list[x]
                inputs_seq_len[i_batch] = frame_num
                labels_seq_len[i_batch] = len(self.label_list[x])
                input_names[i_batch] = basename(
                    self.input_paths[x]).split('.')[0]

        return inputs, labels, inputs_seq_len, labels_seq_len, input_names
