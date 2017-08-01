#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Load dataset for the multitask CTC model (TIMIT corpus).
   In addition, frame stacking and skipping are used.
   You can use only the single GPU version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import pickle
import numpy as np

from experiments.utils.progressbar import wrap_iterator
from experiments.utils.data.dataset_loader.all_load.multitask_ctc_all_load import DatasetBase
from experiments.utils.data.frame_stacking import stack_frame


class Dataset(DatasetBase):

    def __init__(self, data_type, label_type_main, label_type_sub, batch_size,
                 num_stack=None, num_skip=None,
                 sort_utt=True, sort_stop_epoch=None, progressbar=False):
        """A class for loading dataset.
        Args:
            data_type: string, train or dev or test
            label_type_main: string, character or character_capital_divide
            label_type_sub: string, phone39 or phone48 or phone61
            batch_size: int, the size of mini-batch
            num_stack: int, the number of frames to stack
            num_skip: int, the number of frames to skip
            sort_utt: if True, sort all utterances by the number of frames and
                utteraces in each mini-batch are shuffled
            sort_stop_epoch: After sort_stop_epoch, training will revert back
                to a random order
            progressbar: if True, visualize progressbar
        """
        if data_type not in ['train', 'dev', 'test']:
            raise ValueError('data_type is "train" or "dev" or "test".')
        self.is_training = True if data_type == 'train' else False

        self.data_type = data_type
        self.label_type_main = label_type_main
        self.label_type_sub = label_type_sub
        self.batch_size = batch_size
        self.num_stack = num_stack
        self.num_skip = num_skip
        self.sort_utt = sort_utt
        self.sort_stop_epoch = sort_stop_epoch
        self.epoch = 0
        self.progressbar = progressbar

        input_path = join(
            '/n/sd8/inaguma/corpus/timit/dataset/inputs', data_type)
        label_main_path = join(
            '/n/sd8/inaguma/corpus/timit/dataset/labels/ctc',
            label_type_main, data_type)
        label_sub_path = join(
            '/n/sd8/inaguma/corpus/timit/dataset/labels/ctc',
            label_type_sub, data_type)

        # Load the frame number dictionary
        with open(join(input_path, 'frame_num.pickle'), 'rb') as f:
            self.frame_num_dict = pickle.load(f)

        # Sort paths to input & label by frame num
        frame_num_tuple_sorted = sorted(self.frame_num_dict.items(),
                                        key=lambda x: x[1])
        input_paths, label_main_paths, label_sub_paths = [], [], []
        for input_name, frame_num in frame_num_tuple_sorted:
            input_paths.append(join(input_path, input_name + '.npy'))
            label_main_paths.append(join(label_main_path, input_name + '.npy'))
            label_sub_paths.append(
                join(label_sub_path,  input_name + '.npy'))
        if len(label_main_paths) != len(label_sub_paths):
            raise ValueError('The numbers of labels between ' +
                             'character and phone are not same.')
        self.input_paths = np.array(input_paths)
        self.label_main_paths = np.array(label_main_paths)
        self.label_sub_paths = np.array(label_sub_paths)
        self.data_num = len(self.input_paths)

        # Load all dataset in advance
        print('=> Loading dataset (%s, %s, %s)...' %
              (data_type, label_type_main, label_type_sub))
        input_list, label_main_list, label_sub_list = [], [], []
        for i in wrap_iterator(range(self.data_num), self.progressbar):
            input_list.append(np.load(self.input_paths[i]))
            label_main_list.append(np.load(self.label_main_paths[i]))
            label_sub_list.append(np.load(self.label_sub_paths[i]))
        self.input_list = np.array(input_list)
        self.label_main_list = np.array(label_main_list)
        self.label_sub_list = np.array(label_sub_list)
        self.input_size = self.input_list[0].shape[1]

        # Frame stacking
        if (num_stack is not None) and (num_skip is not None):
            print('=> Stacking frames...')
            self.input_list = stack_frame(self.input_list,
                                          self.input_paths,
                                          self.frame_num_dict,
                                          num_stack,
                                          num_skip,
                                          progressbar)
            self.input_size *= num_stack

        self.rest = set(range(0, self.data_num, 1))
