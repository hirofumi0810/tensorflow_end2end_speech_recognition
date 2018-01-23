#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Load dataset for the CTC model (ERATO corpus).
   In addition, frame stacking and skipping are used.
   You can use only the single GPU version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, isfile
import pickle
import numpy as np

from utils.dataset.ctc import DatasetBase


class Dataset(DatasetBase):

    def __init__(self, data_type, label_type, ss_type, batch_size,
                 max_epoch=None, splice=1,
                 num_stack=1, num_skip=1,
                 shuffle=False, sort_utt=False, sort_stop_epoch=None,
                 progressbar=False):
        """A class for loading dataset.
        Args:
            data_type (string): train or dev or test
            label_type (string): kana
            ss_type (string): remove or insert_left or insert_both or insert_right
            batch_size (int): the size of mini-batch
            max_epoch (int, optional): the max epoch. None means infinite loop.
            splice (int, optional): frames to splice. Default is 1 frame.
            num_stack (int, optional): the number of frames to stack
            num_skip (int, optional): the number of frames to skip
            shuffle (bool, optional): if True, shuffle utterances. This is
                disabled when sort_utt is True.
            sort_utt (bool, optional): if True, sort all utterances by the
                number of frames and utteraces in each mini-batch are shuffled.
                Otherwise, shuffle utteraces.
            sort_stop_epoch (int, optional): After sort_stop_epoch, training
                will revert back to a random order
            progressbar (bool, optional): if True, visualize progressbar
        """
        super(Dataset, self).__init__()

        self.is_test = True if data_type == 'test' else False

        self.data_type = data_type
        self.label_type = label_type
        self.ss_type = ss_type
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.splice = splice
        self.num_stack = num_stack
        self.num_skip = num_skip
        self.shuffle = shuffle
        self.sort_utt = sort_utt
        self.sort_stop_epoch = sort_stop_epoch
        self.progressbar = progressbar
        self.num_gpu = 1

        # paths where datasets exist
        dataset_root = ['/data/inaguma/erato',
                        '/n/sd8/inaguma/corpus/erato/dataset']

        input_path = join(dataset_root[0], 'inputs', data_type)
        # NOTE: ex.) save_path:
        # erato_dataset_path/inputs/data_type/speaker/***.npy
        label_path = join(dataset_root[0], 'labels',
                          ss_type, data_type, label_type)
        # NOTE: ex.) save_path:
        # erato_dataset_path/labels/ss_type/data_type/kana/speaker/***.npy

        # Load the frame number dictionary
        if isfile(join(input_path, 'frame_num.pickle')):
            with open(join(input_path, 'frame_num.pickle'), 'rb') as f:
                self.frame_num_dict = pickle.load(f)
        else:
            dataset_root.pop(0)
            input_path = join(dataset_root[0], 'inputs', data_type)
            label_path = join(dataset_root[0], 'labels',
                              ss_type, data_type, label_type)
            with open(join(input_path, 'frame_num.pickle'), 'rb') as f:
                self.frame_num_dict = pickle.load(f)

        # Sort paths to input & label
        axis = 1 if sort_utt else 0
        frame_num_tuple_sorted = sorted(self.frame_num_dict.items(),
                                        key=lambda x: x[axis])
        input_paths, label_paths = [], []
        for input_name, frame_num in frame_num_tuple_sorted:
            speaker = '_'.join(input_name.split('_')[:-1])
            input_paths.append(join(input_path, speaker, input_name + '.npy'))
            label_paths.append(join(label_path, speaker, input_name + '.npy'))
        self.input_paths = np.array(input_paths)
        self.label_paths = np.array(label_paths)
        # NOTE: Not load dataset yet

        self.rest = set(range(0, len(self.input_paths), 1))
