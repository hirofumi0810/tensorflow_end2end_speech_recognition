#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Load dataset for the multitask CTC model.
   You can use the multi-GPU version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import tensorflow as tf

from .frame_stack import stack_frame
from ..sparsetensor import list2sparsetensor


class DatasetBase(object):

    def __init__(self, data_type, label_type_main, label_type_second,
                 batch_size, num_stack=None, num_skip=None,
                 is_sorted=True, is_progressbar=False, num_gpu=1):
        """Load mini-batch in each step.
        Args:
            data_type: string
            label_type_main: string
            label_type_second: string
            batch_size: int, the size of mini-batch
            num_stack: int, the number of frames to stack
            num_skip: int, the number of frames to skip
            is_sorted: if True, sort dataset by frame num
            is_progressbar: if True, visualize progressbar
            num_gpu: int, if more than 1, divide batch_size by num_gpu
        """
        self.data_type = data_type
        self.label_type_main = label_type_main
        self.label_type_second = label_type_second
        self.batch_size = batch_size * num_gpu
        self.is_sorted = is_sorted
        self.is_progressbar = is_progressbar
        self.num_gpu = num_gpu

        self.input_size = None
        self.dataset_path = None

        # Step
        # 1. Load the frame number dictionary
        self.frame_num_dict = None

        # 2. Load all paths to input & label
        self.input_paths = None
        self.label_main_paths = None
        self.label_second_paths = None
        self.data_num = None

        # 3. Load all dataset in advance
        self.input_list = None
        self.label_main_list = None
        self.label_second_list = None
        self.rest = set([i for i in range(self.data_num)])


def next_batch(self, batch_size=None, session=None):
    """Make mini-batch.
    Args:
        batch_size: int, the size of mini-batch
        session:
    Returns:
        inputs: list of input data, size `[batch_size]`
        labels_main_st: list of SparseTensor of labels in the main task
            if num_gpu > 1, list of labels_st, size of num_gpu
        labels_second_st: list of SparseTensor of labels in the second task
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
                self.rest = set(
                    [i for i in range(self.data_num)])
                next_epoch_flag = True
                if self.data_type == 'train':
                    print('---Next epoch---')

            # Shuffle selected mini-batch
            random.shuffle(sorted_indices)

            # Load dataset in mini-batch
            input_list, label_main_list = [], []
            label_second_list, input_name_list = [], []
            for i in sorted_indices:
                # input_list.append(np.load(self.input_paths[i]))
                # label_main_list.append(np.load(self.label_main_paths[i]))
                # label_second_list.append(np.load(self.label_second_paths[i]))
                # input_name_list.append(os.path.basename(
                #     self.input_paths[i]).split('.')[0])
                input_list.append(np.load(np.take(self.input_paths, i,
                                                  axis=0)))
                label_main_list.append(
                    np.load(np.take(self.label_main_paths, i, axis=0)))
                label_second_list.append(
                    np.load(np.take(self.label_second_paths, i, axis=0)))
                input_name_list.append(
                    os.path.basename(np.take(self.input_paths, i,
                                             axis=0)).split('.')[0])
            input_list = np.array(input_list)
            label_main_list = np.array(label_main_list)
            label_second_list = np.array(label_second_list)
            input_name_list = np.array(input_name_list)

            # Frame stacking
            if (self.num_stack is not None) and (self.num_skip is not None):
                stacked_input_list = stack_frame(
                    input_list,
                    self.input_paths[sorted_indices],
                    self.frame_num_dict,
                    self.num_stack,
                    self.num_skip,
                    is_progressbar=False)
                input_list = np.array(stacked_input_list)

            # Compute max frame num in mini-batch
            max_frame_num = max(map(lambda x: x.shape[0], input_list))

            # Compute max target label length in mini-batch
            max_seq_len_main = max(map(len, label_main_list))
            max_seq_len_second = max(map(len, label_second_list))

            # Initialization
            inputs = np.zeros(
                (len(sorted_indices), max_frame_num, self.input_size))
            # Padding with -1
            labels_main = np.array([[-1] * max_seq_len_main]
                                   * len(sorted_indices), dtype=int)
            labels_second = np.array([[-1] * max_seq_len_second]
                                     * len(sorted_indices), dtype=int)
            inputs_seq_len = np.empty((len(sorted_indices),), dtype=int)
            input_names = [None] * len(sorted_indices)

            # Set values of each data in mini-batch
            for i_batch in range(len(sorted_indices)):
                data_i = input_list[i_batch]
                frame_num = data_i.shape[0]
                inputs[i_batch, :frame_num, :] = data_i
                labels_main[i_batch, :len(
                    label_main_list[i_batch])] = label_main_list[i_batch]
                labels_second[i_batch, :len(
                    label_second_list[i_batch])] = label_second_list[i_batch]
                inputs_seq_len[i_batch] = frame_num
                input_names[i_batch] = input_name_list[i_batch]

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

            # Load dataset in mini-batch
            input_list, label_main_list = [], []
            label_second_list, input_name_list = [], []
            for i in random_indices:
                # input_list.append(np.load(self.input_paths[i]))
                # label_main_list.append(np.load(self.label_main_paths[i]))
                # label_second_list.append(np.load(self.label_second_paths[i]))
                # input_name_list.append(
                #     os.path.basename(self.input_paths[i]).split('.')[0])
                input_list.append(
                    np.load(np.take(self.input_paths, i, axis=0)))
                label_main_list.append(
                    np.load(np.take(self.label_main_paths, i, axis=0)))
                label_second_list.append(
                    np.load(np.take(self.label_second_paths, i, axis=0)))
                input_name_list.append(os.path.basename(
                    np.take(self.input_paths, i, axis=0)).split('.')[0])
            input_list = np.array(input_list)
            label_main_list = np.array(label_main_list)
            label_second_list = np.array(label_second_list)
            input_name_list = np.array(input_name_list)

            # Frame stacking
            if (self.num_stack is not None) and (self.num_skip is not None):
                stacked_input_list = stack_frame(
                    input_list,
                    self.input_paths[random_indices],
                    self.frame_num_dict,
                    self.num_stack,
                    self.num_skip,
                    is_progressbar=False)
                input_list = np.array(stacked_input_list)

            # Compute max frame num in mini-batch
            max_frame_num = max(map(lambda x: x.shape[0], input_list))

            # Compute max target label length in mini-batch
            max_seq_len_main = max(map(len, label_main_list))
            max_seq_len_second = max(map(len, label_second_list))

            # Initialization
            inputs = np.zeros(
                (len(random_indices), max_frame_num, self.input_size))
            # Padding with -1
            labels_main = np.array([[-1] * max_seq_len_main]
                                   * len(random_indices), dtype=int)
            labels_second = np.array([[-1] * max_seq_len_second]
                                     * len(random_indices), dtype=int)
            inputs_seq_len = np.empty((len(random_indices),), dtype=int)
            input_names = [None] * len(random_indices)

            # Set values of each data in mini-batch
            for i_batch in range(len(random_indices)):
                data_i = input_list[i_batch]
                frame_num = data_i.shape[0]
                inputs[i_batch, : frame_num, :] = data_i
                labels_main[i_batch, :len(
                    label_main_list[i_batch])] = label_main_list[i_batch]
                labels_second[i_batch, :len(
                    label_second_list[i_batch])] = label_second_list[i_batch]
                inputs_seq_len[i_batch] = frame_num
                input_names[i_batch] = input_name_list[i_batch]

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
            labels_main = tf.split(labels_main, divide_num, axis=0)
            labels_second = tf.split(labels_second, divide_num, axis=0)
            inputs_seq_len = tf.split(inputs_seq_len, divide_num, axis=0)
            input_names = tf.split(input_names, divide_num, axis=0)

            # Convert from SparseTensor to numpy.ndarray
            inputs = list(map(session.run, inputs))
            labels_char = list(map(session.run, labels_main))
            labels_phone = list(map(session.run, labels_second))
            labels_main_st = list(map(list2sparsetensor, labels_char))
            labels_second_st = list(map(list2sparsetensor, labels_phone))
            inputs_seq_len = list(map(session.run, inputs_seq_len))
            input_names = list(map(session.run, input_names))
        else:
            labels_main_st = list2sparsetensor(labels_main)
            labels_second_st = list2sparsetensor(labels_second)

        yield (inputs, labels_main_st, labels_second_st, inputs_seq_len,
               input_names)
