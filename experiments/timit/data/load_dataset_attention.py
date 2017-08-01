#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Load dataset for the Attention model (TIMIT corpus).
   You can use only the single GPU version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import pickle
import numpy as np

from experiments.utils.progressbar import wrap_iterator
from experiments.utils.data.dataset_loader.all_load.attention_all_load import DatasetBase


class Dataset(DatasetBase):

    def __init__(self, data_type, label_type, batch_size, eos_index,
                 sort_utt=True, sort_stop_epoch=None, progressbar=False):
        """A class for loading dataset.
        Args:
            data_type: string, train or dev or test
            label_type: string, phone39 or phone48 or phone61 or character
                or character_capital_divide
            eos_index: int , the index of <EOS> class
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
        self.label_type = label_type
        self.batch_size = batch_size
        self.eos_index = eos_index
        self.sort_utt = sort_utt
        self.sort_stop_epoch = sort_stop_epoch
        self.epoch = 0
        self.progressbar = progressbar

        input_path = join(
            '/n/sd8/inaguma/corpus/timit/dataset/inputs', data_type)
        label_path = join(
            '/n/sd8/inaguma/corpus/timit/dataset/labels/attention',
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
        print('=> Loading dataset (%s, %s)...' % (data_type, label_type))
        input_list, label_list = [], []
        for i in wrap_iterator(range(self.data_num), self.progressbar):
            input_list.append(np.load(self.input_paths[i]))
            label_list.append(np.load(self.label_paths[i]))
        self.input_list = np.array(input_list)
        self.label_list = np.array(label_list)
        self.input_size = self.input_list[0].shape[1]

        self.rest = set(range(0, self.data_num, 1))
