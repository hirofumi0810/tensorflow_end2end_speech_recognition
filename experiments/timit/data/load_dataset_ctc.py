#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Load dataset for the CTC model (TIMIT corpus).
   In addition, frame stacking and skipping are used.
   You can use the multi-GPU version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import pickle
import numpy as np

from experiments.utils.progressbar import wrap_iterator
from experiments.utils.data.dataset_loader.all_load.ctc_all_load import DatasetBase
from experiments.utils.data.frame_stacking import stack_frame


class Dataset(DatasetBase):

    def __init__(self, data_type, label_type, batch_size,
                 num_stack=None, num_skip=None,
                 sort_utt=True, progressbar=False, num_gpu=1):
        """A class for loading dataset.
        Args:
            data_type: string, train or dev or test
            label_type: stirng, phone39 or phone48 or phone61 or character
            batch_size: int, the size of mini-batch
            num_stack: int, the number of frames to stack
            num_skip: int, the number of frames to skip
            sort_utt: if True, sort all utterances by the number of frames
            progressbar: if True, visualize progressbar
            num_gpu: int, if more than 1, divide batch_size by num_gpu
        """
        if data_type not in ['train', 'dev', 'test']:
            raise ValueError('data_type is "train" or "dev" or "test".')

        self.data_type = data_type
        self.label_type = label_type
        self.batch_size = batch_size * num_gpu
        self.num_stack = num_stack
        self.num_skip = num_skip
        self.sort_utt = sort_utt
        self.progressbar = progressbar
        self.num_gpu = num_gpu

        self.input_size = 123
        input_path = join(
            '/n/sd8/inaguma/corpus/timit/dataset/inputs/', data_type)
        label_path = join(
            '/n/sd8/inaguma/corpus/timit/dataset/labels/ctc/',
            label_type, data_type)

        # Load the frame number dictionary
        with open(join(input_path, 'frame_num.pickle'), 'rb') as f:
            self.frame_num_dict = pickle.load(f)

        # Sort paths to input & label by frame num
        frame_num_tuple_sorted = sorted(self.frame_num_dict.items(),
                                        key=lambda x: x[1])
        input_paths, label_paths = [], []
        for input_name, frame_num in frame_num_tuple_sorted:
            input_paths.append(join(input_path, input_name + '.npy'))
            label_paths.append(join(label_path, input_name + '.npy'))
        self.input_paths = np.array(input_paths)
        self.label_paths = np.array(label_paths)
        self.data_num = len(self.input_paths)

        # Load all dataset in advance
        print('=> Loading ' + data_type + ' dataset (' + label_type + ')...')
        input_list, label_list = [], []
        for i in wrap_iterator(range(self.data_num), self.progressbar):
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
                                             progressbar)
            self.input_list = np.array(stacked_input_list)
            self.input_size = self.input_size * num_stack

        self.rest = set(range(0, self.data_num, 1))
