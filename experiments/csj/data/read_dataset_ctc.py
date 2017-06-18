#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Read dataset for CTC network (CSJ corpus).
   In addition, frame stacking and skipping are used.
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
        if batch_size % num_gpu != 0:
            raise ValueError('batch_size shoule be by a factor of num_gpu.')

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
        self.input_size = self.input_size * self.num_stack
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
            self.frame_num_dict.items(), key=lambda x: x[1])
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
        if train_data_size == 'default':
            self.num_cluster = 10
        elif train_data_size == 'large':
            self.num_cluster = 15
        if data_type in ['train', 'train_all']:
            self.rest_cluster = self.num_cluster - 1
            self.data_num_cluster = int(
                (self.data_num / self.num_cluster) / 128) * 128
            self.input_paths_cluster = self.input_paths[
                0:self.data_num_cluster]
            self.label_paths_cluster = self.label_paths[
                0:self.data_num_cluster]
        else:
            self.rest_cluster = 0
            self.data_num_cluster = self.data_num
            self.input_paths_cluster = self.input_paths
            self.label_paths_cluster = self.label_paths

        # Load dataset in one cluster
        self.next_cluster()
        self.next_cluster_flag = False

    def next_cluster(self):
        # Load all dataset
        print('=> Loading next cluster...')
        self.input_list, self.label_list = [], []
        for i in wrap_iterator(range(self.data_num_cluster),
                               self.is_progressbar):
            self.input_list.append(
                np.load(self.input_paths_cluster[i]))
            self.label_list.append(np.load(self.label_paths_cluster[i]))
        self.input_list = np.array(self.input_list)
        self.label_list = np.array(self.label_list)

        # Frame stacking
        if (self.num_stack is not None) and (self.num_skip is not None):
            print('=> Stacking frames...')
            stacked_input_list = stack_frame(self.input_list,
                                             self.input_paths_cluster,
                                             self.frame_num_dict,
                                             self.num_stack,
                                             self.num_skip,
                                             self.is_progressbar)
            self.input_list = np.array(stacked_input_list)

        self.rest = set([j for j in range(len(self.input_paths_cluster))])

    def next_batch(self, batch_size=None, session=None):
        """Make mini batch.
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

        #########################
        # sorted dataset
        #########################
        if self.is_sorted:
            if len(self.rest) > self.batch_size:
                sorted_indices = list(self.rest)[:self.batch_size]
                self.rest -= set(sorted_indices)
            else:
                sorted_indices = list(self.rest)
                if self.data_type == 'train':
                    self.next_cluster_flag = True
                    print('---Next cluster---')
                else:
                    self.rest = set(
                        [i for i in range(len(self.input_paths_cluster))])

            # Compute max frame num in mini batch
            max_frame_num = self.input_list[sorted_indices[-1]].shape[0]

            # Compute max target label length in mini batch
            max_seq_len = max(map(len, self.label_list[sorted_indices]))

            # Shuffle selected mini batch (0 ~ len(self.rest)-1)
            random.shuffle(sorted_indices)

            # Initialization
            inputs = np.zeros(
                (len(sorted_indices), max_frame_num, self.input_size))
            labels = np.array([[-1] * max_seq_len]  # Padding by -1
                              * len(sorted_indices), dtype=int)
            inputs_seq_len = np.empty((len(sorted_indices),), dtype=int)
            input_names = [None] * len(sorted_indices)

            # Set values of each data in mini batch
            for i_batch, x in enumerate(sorted_indices):
                data_i = self.input_list[x]
                frame_num = data_i.shape[0]
                inputs[i_batch, :frame_num, :] = data_i
                labels[i_batch, :len(self.label_list[x])] = self.label_list[x]
                inputs_seq_len[i_batch] = frame_num
                input_names[i_batch] = basename(
                    self.input_paths_cluster[x]).split('.')[0]

            if self.next_cluster_flag:
                if self.rest_cluster >= 1:
                    # Set fot the next clusters
                    frame_offset = (self.num_cluster -
                                    self.rest_cluster) * self.data_num_cluster
                    self.input_paths_cluster = self.input_paths[
                        frame_offset:frame_offset + self.data_num_cluster]
                    self.label_paths_cluster = self.label_paths[
                        frame_offset: frame_offset + self.data_num_cluster]
                    self.rest_cluster -= 1
                else:
                    # Initialize clusters
                    if self.data_type == 'train':
                        self.rest_cluster = self.num_cluster - 1
                        self.input_paths_cluster = self.input_paths[
                            0: self.data_num_cluster]
                        self.label_paths_cluster = self.label_paths[
                            0: self.data_num_cluster]
                        print('---Next epoch---')

                # Load dataset in the next cluster
                self.next_cluster()
                self.next_cluster_flag = False

        #########################
        # not sorted dataset
        #########################
        else:
            if len(self.rest) > self.batch_size:
                # Randomly sample mini batch
                random_indices = random.sample(
                    list(self.rest), self.batch_size)
                self.rest -= set(random_indices)
            else:
                random_indices = list(self.rest)
                self.rest = set(
                    [i for i in range(len(self.input_paths_cluster))])
                if self.data_type == 'train':
                    print('---Next epoch---')

                # Shuffle selected mini batch (0 ~ len(self.rest)-1)
                random.shuffle(random_indices)

            # Compute max frame num in mini batch
            max_frame_num = max(
                map(lambda x: x.shape[0], self.input_list[random_indices]))

            # Compute max target label length in mini batch
            max_seq_len = max(map(len, self.label_list[random_indices]))

            # Initialization
            inputs = np.zeros(
                (len(random_indices), max_frame_num, self.input_size))
            labels = np.array([[-1] * max_seq_len]  # Padding by -1
                              * len(random_indices), dtype=int)
            inputs_seq_len = np.empty((len(random_indices),), dtype=int)
            input_names = [None] * len(random_indices)

            # Set values of each data in mini batch
            for i_batch, x in enumerate(random_indices):
                data_i = self.input_list[x]
                frame_num = data_i.shape[0]
                inputs[i_batch, : frame_num, :] = data_i
                labels[i_batch, :len(self.label_list[x])] = self.label_list[x]
                inputs_seq_len[i_batch] = frame_num
                input_names[i_batch] = basename(
                    self.input_paths_cluster[x]).split('.')[0]

        if self.num_gpu > 1:
            # Now we split the mini-batch data by num_gpu
            inputs = tf.split(inputs, self.num_gpu, axis=0)
            labels = tf.split(labels, self.num_gpu, axis=0)
            inputs_seq_len = tf.split(inputs_seq_len, self.num_gpu, axis=0)
            input_names = tf.split(input_names, self.num_gpu, axis=0)

            labels_st = []
            for i_gpu in range(self.num_gpu):
                labels_st.append(list2sparsetensor(
                    session.run(labels[i_gpu])))
        else:
            labels_st = list2sparsetensor(labels)

        return inputs, labels_st, inputs_seq_len, input_names
