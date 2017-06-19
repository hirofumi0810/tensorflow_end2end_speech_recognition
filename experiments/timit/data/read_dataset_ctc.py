#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Read dataset for CTC network (TIMIT corpus).
   In addition, frame stacking and skipping are used.
   You can use the multi-GPU version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, basename
import pickle
import random
import numpy as np
import tensorflow as tf

from utils.frame_stack import stack_frame
from utils.sparsetensor import list2sparsetensor
from utils.progressbar import wrap_iterator


class DataSet(object):
    """Read dataset."""

    def __init__(self, data_type, label_type, batch_size,
                 num_stack=None, num_skip=None,
                 is_sorted=True, is_progressbar=False, num_gpu=1):
        """
        Args:
            data_type: string, train or dev or test
            label_type: stirng, phone39 or phone48 or phone61 or character
            batch_size: int, the size of mini-batch
            num_stack: int, the number of frames to stack
            num_skip: int, the number of frames to skip
            is_sorted: if True, sort dataset by frame num
            is_progressbar: if True, visualize progressbar
            num_gpu: int, if more than 1, divide batch_size by num_gpu
        """
        if data_type not in ['train', 'dev', 'test']:
            raise ValueError('data_type is "train" or "dev" or "test".')

        self.data_type = data_type
        self.label_type = label_type
        self.batch_size = batch_size * num_gpu
        self.num_stack = num_stack
        self.num_skip = num_skip
        self.is_sorted = is_sorted
        self.is_progressbar = is_progressbar
        self.num_gpu = num_gpu

        self.input_size = 123
        self.dataset_path = join(
            '/n/sd8/inaguma/corpus/timit/dataset/ctc/', label_type, data_type)

        # Load the frame number dictionary
        self.frame_num_dict_path = join(self.dataset_path, 'frame_num.pickle')
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

        # Load all dataset in advance
        print('=> Loading ' + data_type + ' dataset (' + label_type + ')...')
        input_list, label_list = [], []
        for i in wrap_iterator(range(self.data_num), self.is_progressbar):
            input_list.append(np.load(self.input_paths[i]))
            label_list.append(np.load(self.label_paths[i]))
        self.input_list = np.array(input_list)
        self.label_list = np.array(label_list)

        # Frame stacking
        if (num_stack is not None) and (num_skip is not None):
            print('=> Stacking frames...')
            stacked_input_list = stack_frame(self.input_list,
                                             self.input_paths,
                                             self.frame_num_dict,
                                             num_stack,
                                             num_skip,
                                             is_progressbar)
            self.input_list = np.array(stacked_input_list)
            self.input_size = self.input_size * num_stack

        self.rest = set([i for i in range(self.data_num)])

    def next_batch(self, batch_size=None, session=None):
        """Make mini-batch.
        Args:
            batch_size: int, the size of mini-batch
            session:
        Returns:
            inputs: list of input data, size `[batch_size]`
            labels_st: list of SparseTensor of labels
                if num_gpu > 1, list of labels_st, size of num_gpu
            inputs_seq_len: list of length of inputs of size `[batch_size]`
            input_names: list of file name of input data of size `[batch_size]`
        """
        if session is None and self.num_gpu != 1:
            raise ValueError('Set session when using multiple GPUs.')

        if batch_size is None:
            batch_size = self.batch_size

        next_epoch_flag = False

        while True:
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
                    next_epoch_flag = True
                    if self.data_type == 'train':
                        print('---Next epoch---')

                # Compute max frame num in mini-batch
                max_frame_num = self.input_list[sorted_indices[-1]].shape[0]

                # Compute max target label length in mini-batch
                max_seq_len = max(map(len, self.label_list[sorted_indices]))

                # Shuffle selected mini-batch
                random.shuffle(sorted_indices)

                # Initialization
                inputs = np.zeros(
                    (len(sorted_indices), max_frame_num, self.input_size))
                # Padding with -1
                labels = np.array([[-1] * max_seq_len]
                                  * len(sorted_indices), dtype=int)
                inputs_seq_len = np.empty((len(sorted_indices),), dtype=int)
                input_names = [None] * len(sorted_indices)

                # Set values of each data in mini-batch
                for i_batch, x in enumerate(sorted_indices):
                    data_i = self.input_list[x]
                    frame_num = data_i.shape[0]
                    inputs[i_batch, :frame_num, :] = data_i
                    labels[i_batch, :len(self.label_list[x])
                           ] = self.label_list[x]
                    inputs_seq_len[i_batch] = frame_num
                    input_names[i_batch] = basename(
                        self.input_paths[x]).split('.')[0]

            #########################
            # not sorted dataset
            #########################
            else:
                if len(self.rest) > batch_size:
                    # Randomly sample mini-batch
                    random_indices = random.sample(
                        list(self.rest), batch_size)
                    self.rest -= set(random_indices)
                else:
                    random_indices = list(self.rest)
                    self.rest = set([i for i in range(self.data_num)])
                    next_epoch_flag = True
                    if self.data_type == 'train':
                        print('---Next epoch---')

                    # Shuffle selected mini-batch
                    random.shuffle(random_indices)

                # Compute max frame num in mini-batch
                max_frame_num = max(
                    map(lambda x: x.shape[0], self.input_list[random_indices]))

                # Compute max target label length in mini-batch
                max_seq_len = max(map(len, self.label_list[random_indices]))

                # Initialization
                inputs = np.zeros(
                    (len(random_indices), max_frame_num, self.input_size))
                # Padding with -1
                labels = np.array([[-1] * max_seq_len]
                                  * len(random_indices), dtype=int)
                inputs_seq_len = np.empty((len(random_indices),), dtype=int)
                input_names = [None] * len(random_indices)

                # Set values of each data in mini-batch
                for i_batch, x in enumerate(random_indices):
                    data_i = self.input_list[x]
                    frame_num = data_i.shape[0]
                    inputs[i_batch, :frame_num, :] = data_i
                    labels[i_batch, :len(self.label_list[x])
                           ] = self.label_list[x]
                    inputs_seq_len[i_batch] = frame_num
                    input_names[i_batch] = basename(
                        self.input_paths[x]).split('.')[0]

            if self.num_gpu > 1:
                divide_num = self.num_gpu
                if next_epoch_flag:
                    for i in range(self.num_gpu, 0, -1):
                        if len(self.rest) % i == 0:
                            divide_num = i
                            break
                    next_epoch_flag = False

                # Now we split the mini-batch data by num_gpu
                inputs = tf.split(inputs, divide_num, axis=0)
                labels = tf.split(labels, divide_num, axis=0)
                inputs_seq_len = tf.split(inputs_seq_len, divide_num, axis=0)
                input_names = tf.split(input_names, divide_num, axis=0)

                # Convert from SparseTensor to numpy.ndarray
                inputs = list(map(session.run, inputs))
                labels = list(map(session.run, labels))
                labels_st = list(map(list2sparsetensor, labels))
                inputs_seq_len = list(map(session.run, inputs_seq_len))
                input_names = list(map(session.run, input_names))
            else:
                labels_st = list2sparsetensor(labels)

            yield inputs, labels_st, inputs_seq_len, input_names
