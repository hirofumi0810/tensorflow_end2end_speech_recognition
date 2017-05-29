#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Read dataset for CTC network (TIMIT corpus).
   In addition, frame stacking and skipping are used.
"""

import os
import sys
import pickle
import random
import numpy as np
from tqdm import tqdm

sys.path.append('../../')
from utils.data.frame_stack import stack_frame


class DataSet(object):
    """Read dataset."""

    def __init__(self, data_type, label_type, num_stack=None, num_skip=None, is_sorted=True, is_progressbar=False):
        """
        Args:
            data_type: train or dev or test
            label_type: phone39 or phone48 or phone61 or character
            num_stack: int, the number of frames to stack
            num_skip: int, the number of frames to skip
            is_sorted: if True, sort dataset by frame num
            is_progressbar: if True, visualize progressbar
        """
        if data_type not in ['train', 'dev', 'test']:
            raise ValueError('Error: data_type is "train" or "dev" or "test".')

        self.data_type = data_type
        self.label_type = label_type
        self.num_stack = num_stack
        self.num_skip = num_skip
        self.is_sorted = is_sorted
        self.is_progressbar = is_progressbar

        self.input_size = 123
        self.dataset_path = os.path.join(
            '/n/sd8/inaguma/corpus/timit/dataset/ctc/', label_type, data_type)

        # Load the frame number dictionary
        self.frame_num_dict_path = os.path.join(
            self.dataset_path, 'frame_num.pickle')
        with open(self.frame_num_dict_path, 'rb') as f:
            self.frame_num_dict = pickle.load(f)

        # Sort paths to input & label by frame num
        self.frame_num_tuple_sorted = sorted(
            self.frame_num_dict.items(), key=lambda x: x[1])
        input_paths, label_paths = [], []
        for input_name, frame_num in self.frame_num_tuple_sorted:
            input_paths.append(os.path.join(
                self.dataset_path, 'input', input_name + '.npy'))
            label_paths.append(os.path.join(
                self.dataset_path, 'label', input_name + '.npy'))
        self.input_paths = np.array(input_paths)
        self.label_paths = np.array(label_paths)
        self.data_num = len(self.input_paths)

        # Setting for progressbar
        iterator = tqdm(range(self.data_num)
                        ) if is_progressbar else range(self.data_num)

        # Load all dataset
        print('=> Loading ' + data_type + ' dataset (' + label_type + ')...')
        input_list, label_list = [], []
        for i in iterator:
            input_list.append(np.load(self.input_paths[i]))
            label_list.append(np.load(self.label_paths[i]))
        self.input_list = np.array(input_list)
        self.label_list = np.array(label_list)

        # Frame stacking
        if (num_stack is not None) and (num_skip is not None):
            print('=> Stacking frames...')
            stacked_input_list = stack_frame(
                self.input_list, self.input_paths, self.frame_num_dict, num_stack, num_skip, is_progressbar)
            self.input_list = np.array(stacked_input_list)
            self.input_size = self.input_size * num_stack

        self.rest = set([i for i in range(len(self.input_paths))])

    def next_batch(self, batch_size):
        """Make mini batch.
        Args:
            batch_size: mini batch size
        Returns:
            input_data: input data for mini batch
            label: tuple of (indices, values, shape)
            seq_len: length of each label in mini batch
            input_names: list of file name of input data
        """
        #########################
        # sorted dataset
        #########################
        if self.is_sorted:
            if len(self.rest) > batch_size:
                sorted_indices = list(self.rest)[:batch_size]
                self.rest -= set(sorted_indices)

            else:
                sorted_indices = list(self.rest)
                self.rest = set([i for i in range(len(self.input_paths))])
                if self.data_type == 'train':
                    print('---Next epoch---')

            # Compute max frame num in mini batch
            max_frame_num = self.input_list[sorted_indices[-1]].shape[0]

            # Shuffle selected mini batch (0 ~ len(self.rest)-1)
            random.shuffle(sorted_indices)

            # Initialization
            input_data = np.zeros(
                (len(sorted_indices), max_frame_num, self.input_size))
            labels = [None] * len(sorted_indices)
            seq_len = np.empty((len(sorted_indices),))
            input_names = [None] * len(sorted_indices)

            # Set values of each data in mini batch
            for i_batch, x in enumerate(sorted_indices):
                data_i = self.input_list[x]
                frame_num = data_i.shape[0]
                input_data[i_batch, :frame_num, :] = data_i
                labels[i_batch] = self.label_list[x]
                seq_len[i_batch] = frame_num
                input_names[i_batch] = os.path.basename(
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
                self.rest = set([i for i in range(len(self.input_paths))])
                if self.data_type == 'train':
                    print('---Next epoch---')

                # Shuffle selected mini batch (0 ~ len(self.rest)-1)
                random.shuffle(random_indices)

            # Compute max frame num in mini batch
            frame_num_list = []
            for data_i in self.input_list[random_indices]:
                frame_num_list.append(data_i.shape[0])
            max_frame_num = max(frame_num_list)

            # Initialization
            input_data = np.zeros(
                (len(random_indices), max_frame_num, self.input_size))
            labels = [None] * len(random_indices)
            seq_len = np.empty((len(random_indices),))
            input_names = [None] * len(random_indices)

            # Set values of each data in mini batch
            for i_batch, x in enumerate(random_indices):
                data_i = self.input_list[x]
                frame_num = data_i.shape[0]
                input_data[i_batch, :frame_num, :] = data_i
                labels[i_batch] = self.label_list[x]
                seq_len[i_batch] = frame_num
                input_names[i_batch] = os.path.basename(
                    self.input_paths[x]).split('.')[0]

        return input_data, labels, seq_len, input_names


def test():
    import sys
    import re
    sys.path.append('../')
    sys.path.append('../../../')
    from utils.labels.character import num2char
    from utils.data.sparsetensor import list2sparsetensor

    dataset = DataSet(data_type='dev', label_type='character',
                      num_stack=3, num_skip=3,
                      is_sorted=True, is_progressbar=True)
    print('=> Reading mini-batch...')
    map_file_path = '../../../../asr_preprocessing/asr_preprocessing/corpus/timit/labels/ctc/char2num.txt'
    for i in tqdm(range(20000)):
        inputs, labels, seq_len, input_names = dataset.next_batch(
            batch_size=64)
        indices, values, dense_shape = list2sparsetensor(labels)

        str_true = num2char(labels[0], map_file_path)
        str_true = re.sub(r'_', ' ', str_true)
        # print(str_true)


if __name__ == '__main__':
    test()
