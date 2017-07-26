#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Load dataset for the CTC model (Librispeech corpus).
   In addition, frame stacking and skipping are used.
   You can use the multi-GPU version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import pickle
import numpy as np

from experiments.utils.data.dataset_loader.each_load.ctc_each_load import DatasetBase


class Dataset(DatasetBase):

    def __init__(self, data_type, train_data_size, label_type, batch_size,
                 num_stack=None, num_skip=None,
                 sort_utt=True, sorta_grad=False,
                 progressbar=False, num_gpu=1, is_gpu=True):
        """A class for loading dataset.
        Args:
            data_type: string, train_clean100 or train_clean360 or
                train_other500 or dev_clean or dev_other or test_clean or
                test_other
            train_data_size: string, train_clean100 or train_clean360 or
                train_other500
            label_type: string, character or character_capital_divide or word
            batch_size: int, the size of mini-batch
            num_stack: int, the number of frames to stack
            num_skip: int, the number of frames to skip
            sort_utt: if True, sort all utterances by the number of frames and
                utteraces in each mini-batch are shuffled
            sorta_grad: if True, sorting utteraces are conducted only in the
                first epoch (not shuffled in each mini-batch). After the first
                epoch, training will revert back to a random order. If sort_utt
                is also True, it will be False.
            progressbar: if True, visualize progressbar
            num_gpu: int, if more than 1, divide batch_size by num_gpu
            is_gpu: bool, if True, use dataset in the GPU server. This is
                useful when data size is very large and you cannot load all
                dataset at once. Then, you should put dataset on the GPU server
                you will use to reduce data-communication time between servers.
        """
        if data_type not in ['train_clean100', 'train_clean360',
                             'train_other500', 'dev_clean', 'dev_other',
                             'test_clean', 'test_other']:
            raise ValueError(
                'data_type is "train_clean100" or "train_clean360" or ' +
                '"train_other500" or "dev_clean" or "dev_other" or ' +
                '"test_clean" "test_other".')

        self.data_type = data_type
        self.train_data_size = train_data_size
        self.label_type = label_type
        self.batch_size = batch_size * num_gpu
        self.num_stack = num_stack
        self.num_skip = num_skip
        self.sort_utt = sort_utt if not sorta_grad else False
        self.sorta_grad = sorta_grad
        self.progressbar = progressbar
        self.num_gpu = num_gpu
        self.input_size = 120

        if is_gpu:
            # GPU server
            input_path = join('/data/inaguma/librispeech/inputs',
                              train_data_size, data_type)
            label_path = join('/data/inaguma/librispeech/labels/ctc',
                              train_data_size, label_type, data_type)
        else:
            # CPU
            input_path = join(
                '/n/sd8/inaguma/corpus/librispeech/dataset/inputs',
                train_data_size, data_type)
            label_path = join(
                '/n/sd8/inaguma/corpus/librispeech/dataset/labels/ctc',
                train_data_size, label_type, data_type)

        # Load the frame number dictionary
        with open(join(input_path, 'frame_num.pickle'), 'rb') as f:
            self.frame_num_dict = pickle.load(f)

        # Sort paths to input & label by frame num
        frame_num_tuple_sorted = sorted(self.frame_num_dict.items(),
                                        key=lambda x: x[1])
        input_paths, label_paths = [], []
        for input_name, frame_num in frame_num_tuple_sorted:
            speaker_name = input_name.split('-')[0]
            input_paths.append(
                join(input_path, speaker_name, input_name + '.npy'))
            label_paths.append(
                join(label_path, speaker_name, input_name + '.npy'))
        self.input_paths = np.array(input_paths)
        self.label_paths = np.array(label_paths)
        self.data_num = len(self.input_paths)
        print(self.data_num)

        if (self.num_stack is not None) and (self.num_skip is not None):
            self.input_size = self.input_size * num_stack
        # NOTE: Not load dataset yet

        self.rest = set(range(0, self.data_num, 1))

        if data_type in ['test_clean', 'test_other'] and label_type == 'word':
            self.is_test = True
        else:
            self.is_test = False
