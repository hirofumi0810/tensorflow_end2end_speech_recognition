#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Read dataset for CTC network (CSJ corpus).
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
import time

from utils.frame_stack import stack_frame
from utils.sparsetensor import list2sparsetensor
from utils.progressbar import wrap_iterator


class DataSet(object):
    """Read dataset."""

    def __init__(self, data_type, train_data_size, label_type, batch_size,
                 num_stack=None, num_skip=None,
                 is_sorted=True, is_progressbar=False, num_gpu=1):
        """
        Args:
            data_type: string, train, dev, eval1, eval2, eval3
            train_data_size: string, default or large
            label_type: string, phone or character or kanji
            batch_size: int, the size of mini-batch
            num_stack: int, the number of frames to stack
            num_skip: int, the number of frames to skip
            is_sorted: if True, sort dataset by frame num
            is_progressbar: if True, visualize progressbar
            num_gpu: int, if more than 1, divide batch_size by num_gpu
        """
        if data_type not in ['train', 'dev', 'eval1', 'eval2', 'eval3']:
            raise ValueError(
                'data_type is "train" or "dev", "eval1" "eval2" "eval3".')

        self.data_type = data_type * num_gpu
        self.train_data_size = train_data_size
        self.label_type = label_type
        self.batch_size = batch_size * num_gpu
        self.num_stack = num_stack
        self.num_skip = num_skip
        self.is_sorted = is_sorted
        self.is_progressbar = is_progressbar
        self.num_gpu = num_gpu

        self.input_size = 123
        self.input_size = self.input_size
        self.dataset_path = join(
            '/n/sd8/inaguma/corpus/csj/dataset/monolog/ctc/',
            label_type, train_data_size, data_type)

        # Load the frame number dictionary
        self.frame_num_dict_path = join(
            self.dataset_path, 'frame_num.pickle')
        with open(self.frame_num_dict_path, 'rb') as f:
            self.frame_num_dict = pickle.load(f)

        # Sort paths to input & label by frame num
        print('=> loading paths to dataset...')
        self.frame_num_tuple_sorted = sorted(
            self.frame_num_dict.items(), key=lambda x: x[1], reverse=True)
        input_paths, label_paths = [], []
        for input_name, frame_num in wrap_iterator(self.frame_num_tuple_sorted,
                                                   self.is_progressbar):
            speaker_name = input_name.split('_')[0]
            input_paths.append(join(self.dataset_path, 'input',
                                    speaker_name, input_name + '.npy'))
            label_paths.append(join(self.dataset_path, 'label',
                                    speaker_name, input_name + '.npy'))
        self.input_paths = np.array(input_paths)
        self.label_paths = np.array(label_paths)
        self.data_num = len(self.input_paths)

        # Divide dataset into some clusters
        # total: 384198 utterances (train: 240h)
        # total: 896755 utterances (train_all: 586h)

        if (self.num_stack is not None) and (self.num_skip is not None):
            self.input_size = self.input_size * num_stack
        # NOTE: Not load dataset yet

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
                if len(self.rest) > self.batch_size:
                    sorted_indices = list(self.rest)[:self.batch_size]
                    self.rest -= set(sorted_indices)
                else:
                    sorted_indices = list(self.rest)
                    self.rest = set(
                        [i for i in range(len(self.data_num))])
                    next_epoch_flag = True
                    if self.data_type == 'train':
                        print('---Next epoch---')

                # Shuffle selected mini-batch
                random.shuffle(sorted_indices)

                start = time.time()
                # Load dataset in mini-batch
                input_list, label_list, input_name_list = [], [], []
                for i in sorted_indices:
                    # input_list.append(np.load(self.input_paths[i]))
                    # label_list.append(np.load(self.label_paths[i]))
                    # input_name_list.append(basename(
                    #     self.input_paths[i]).split('.')[0])
                    input_list.append(np.load(np.take(self.input_paths, i,
                                                      axis=0)))
                    label_list.append(np.load(np.take(self.label_paths, i,
                                                      axis=0)))
                    input_name_list.append(
                        basename(np.take(self.input_paths, i,
                                         axis=0)).split('.')[0])
                input_list = np.array(input_list)
                label_list = np.array(label_list)
                input_name_list = np.array(input_name_list)
                print('load')
                print(time.time() - start)

                start = time.time()
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
                print('stack')
                print(time.time() - start)

                # Compute max frame num in mini-batch
                max_frame_num = max(map(lambda x: x.shape[0], input_list))

                # Compute max target label length in mini-batch
                max_seq_len = max(map(len, label_list))

                # Initialization
                inputs = np.zeros(
                    (len(sorted_indices), max_frame_num, self.input_size))
                # Padding with -1
                labels = np.array([[-1] * max_seq_len]
                                  * len(sorted_indices), dtype=int)
                inputs_seq_len = np.empty((len(sorted_indices),), dtype=int)
                input_names = [None] * len(sorted_indices)

                # Set values of each data in mini-batch
                for i_batch in range(len(sorted_indices)):
                    data_i = input_list[i_batch]
                    frame_num = data_i.shape[0]
                    inputs[i_batch, :frame_num, :] = data_i
                    labels[i_batch, :len(label_list[i_batch])] = label_list[i_batch]
                    inputs_seq_len[i_batch] = frame_num
                    input_names[i_batch] = input_name_list[i_batch]

            #########################
            # not sorted dataset
            #########################
            else:
                if len(self.rest) > self.batch_size:
                    # Randomly sample mini-batch
                    random_indices = random.sample(
                        list(self.rest), self.batch_size)
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
                input_list, label_list, input_name_list = [], [], []
                for i in random_indices:
                    # input_list.append(np.load(self.input_paths[i]))
                    # label_list.append(np.load(self.label_paths[i]))
                    # input_name_list.append(
                    #     basename(self.input_paths[i]).split('.')[0])
                    input_list.append(np.load(np.take(self.input_paths, i, axis=0)))
                    label_list.append(np.load(np.take(self.label_paths, i, axis=0)))
                    input_name_list.append(
                        basename(np.take(self.input_paths, i, axis=0)).split('.')[0])
                input_list = np.array(input_list)
                label_list = np.array(label_list)
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
                max_seq_len = max(map(len, label_list))

                # Initialization
                inputs = np.zeros(
                    (len(random_indices), max_frame_num, self.input_size))
                # Padding with -1
                labels = np.array([[-1] * max_seq_len]
                                  * len(random_indices), dtype=int)
                inputs_seq_len = np.empty((len(random_indices),), dtype=int)
                input_names = [None] * len(random_indices)

                # Set values of each data in mini-batch
                for i_batch in range(len(random_indices)):
                    data_i = input_list[i_batch]
                    frame_num = data_i.shape[0]
                    inputs[i_batch, : frame_num, :] = data_i
                    labels[i_batch, :len(label_list[i_batch])] = label_list[i_batch]
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

                start = time.time()
                # Now we split the mini-batch data by num_gpu
                inputs = tf.split(inputs, divide_num, axis=0)
                labels = tf.split(labels, divide_num, axis=0)
                inputs_seq_len = tf.split(inputs_seq_len, divide_num, axis=0)
                input_names = tf.split(input_names, divide_num, axis=0)
                print('tf.split')
                print(time.time() - start)

                start = time.time()
                # Convert from SparseTensor to numpy.ndarray
                inputs = list(map(session.run, inputs))
                labels = list(map(session.run, labels))
                labels_st = list(map(list2sparsetensor, labels))
                inputs_seq_len = list(map(session.run, inputs_seq_len))
                input_names = list(map(session.run, input_names))
                print('session.run')
                print(time.time() - start)

            else:
                labels_st = list2sparsetensor(labels)

            yield inputs, labels_st, inputs_seq_len, input_names
