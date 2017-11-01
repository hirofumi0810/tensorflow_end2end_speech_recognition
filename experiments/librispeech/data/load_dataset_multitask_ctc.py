#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Load dataset for the multitask CTC model (Librispeech corpus).
   In addition, frame stacking and skipping are used.
   You can use the multi-GPU version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, isfile
import pickle
import numpy as np

from utils.dataset.multitask_ctc import DatasetBase


class Dataset(DatasetBase):

    def __init__(self, data_type, train_data_size, label_type_main,
                 label_type_sub, batch_size,
                 max_epoch=None, splice=1,
                 num_stack=1, num_skip=1,
                 shuffle=False, sort_utt=False, sort_stop_epoch=None,
                 progressbar=False, num_gpu=1, is_gpu=False):
        """A class for loading dataset.
        Args:
            data_type (stirng): train or dev_clean or dev_other or
                test_clean or test_other
            train_data_size (string): train100h or train460h or train960h
            label_type_main (stirng): word
            label_type_sub (stirng): character or character_capital_divide
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
            num_gpu (int, optional): if more than 1, divide batch_size by num_gpu
        """
        if data_type not in ['train', 'dev_clean', 'dev_other',
                             'test_clean', 'test_other']:
            raise TypeError(
                'data_type is "train" or "dev_clean" or "dev_other" or ' +
                '"test_clean" "test_other".')
        assert 'word' in label_type_main
        if label_type_sub not in ['character', 'character_capital_divide',
                                  'word']:
            raise TypeError(
                'label_type must be "character" or "character_capital_divide" or "word".')

        super(Dataset, self).__init__()

        self.is_training = True if data_type == 'train' else False
        self.is_test = True if 'test' in data_type else False
        # TODO(hirofumi): add phone-level transcripts

        self.data_type = data_type
        self.train_data_size = train_data_size
        self.label_type_main = label_type_main
        self.label_type_sub = label_type_sub
        self.batch_size = batch_size * num_gpu
        self.max_epoch = max_epoch
        self.splice = splice
        self.num_stack = num_stack
        self.num_skip = num_skip
        self.shuffle = shuffle
        self.sort_utt = sort_utt
        self.sort_stop_epoch = sort_stop_epoch
        self.progressbar = progressbar
        self.num_gpu = num_gpu
        self.padded_value = -1

        # paths where datasets exist
        dataset_root = ['/data/inaguma/librispeech',
                        '/n/sd8/inaguma/corpus/librispeech/dataset']

        input_path = join(dataset_root[0], 'inputs',
                          train_data_size, data_type)
        label_main_path = join(dataset_root[0], 'labels',
                               train_data_size, data_type, label_type_main)
        label_sub_path = join(dataset_root[0], 'labels',
                              train_data_size, data_type, label_type_sub)

        # Load the frame number dictionary
        if isfile(join(input_path, 'frame_num.pickle')):
            with open(join(input_path, 'frame_num.pickle'), 'rb') as f:
                self.frame_num_dict = pickle.load(f)
        else:
            dataset_root.pop(0)
            input_path = join(dataset_root[0], 'inputs', train_data_size,
                              data_type)
            label_main_path = join(dataset_root[0], 'labels',
                                   train_data_size, data_type, label_type_main)
            label_sub_path = join(dataset_root[0], 'labels',
                                  train_data_size, data_type, label_type_sub)
            with open(join(input_path, 'frame_num.pickle'), 'rb') as f:
                self.frame_num_dict = pickle.load(f)

        # Sort paths to input & label
        axis = 1 if sort_utt else 0
        frame_num_tuple_sorted = sorted(self.frame_num_dict.items(),
                                        key=lambda x: x[axis])
        input_paths, label_main_paths, label_sub_paths = [], [], []
        for utt_name, frame_num in frame_num_tuple_sorted:
            speaker = utt_name.split('-')[0]
            # ex.) utt_name: speaker-book-utt_index
            input_paths.append(join(input_path, speaker, utt_name + '.npy'))
            label_main_paths.append(
                join(label_main_path, speaker, utt_name + '.npy'))
            label_sub_paths.append(
                join(label_sub_path, speaker, utt_name + '.npy'))
        self.input_paths = np.array(input_paths)
        self.label_main_paths = np.array(label_main_paths)
        self.label_sub_paths = np.array(label_sub_paths)
        # NOTE: Not load dataset yet

        self.rest = set(range(0, len(self.input_paths), 1))
