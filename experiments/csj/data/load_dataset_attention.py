#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Load dataset for the Attention model (CSJ corpus).
   You can use the multi-GPU version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import pickle
import numpy as np

from experiments.utils.progressbar import wrap_iterator
from experiments.utils.data.each_load.attention_each_load import DatasetBase


class Dataset(DatasetBase):

    def __init__(self, data_type, train_data_size, label_type, batch_size,
                 eos_index, sort_utt=True, progressbar=False, num_gpu=1,
                 is_gpu=True, divide_by_space=True):
        """A class for loading dataset.
        Args:
            data_type: string, train, dev, eval1, eval2, eval3
            train_data_size: string, default or large
            label_type: string, kanji or kana or phone
            batch_size: int, the size of mini-batch
            eos_index: int , the index of <EOS> class
            sort_utt: if True, sort all utterances by the number of frames
            progressbar: if True, visualize progressbar
            num_gpu: int, if more than 1, divide batch_size by num_gpu
            is_gpu: bool,
            divide_by_space: if True, each subword will be diveded by space
        """
        if data_type not in ['train', 'dev', 'eval1', 'eval2', 'eval3']:
            raise ValueError(
                'data_type is "train" or "dev", "eval1" "eval2" "eval3".')

        self.data_type = data_type
        self.train_data_size = train_data_size
        self.label_type = label_type
        self.batch_size = batch_size * num_gpu
        self.eos_index = eos_index
        self.sort_utt = sort_utt
        self.progressbar = progressbar
        self.num_gpu = num_gpu
        self.input_size = 123

        if is_gpu:
            # GPU server
            input_path = join('/data/inaguma/csj/inputs',
                              train_data_size, data_type)
            if divide_by_space:
                label_path = join('/data/inaguma/csj/labels/attention_divide/',
                                  train_data_size, label_type, data_type)
            else:
                label_path = join('/data/inaguma/csj/labels/attention/',
                                  train_data_size, label_type, data_type)
        else:
            # CPU
            input_path = join('/n/sd8/inaguma/corpus/csj/dataset/inputs',
                              train_data_size, data_type)
            if divide_by_space:
                label_path = join(
                    '/n/sd8/inaguma/corpus/csj/dataset/labels/attention_divide/',
                    train_data_size, label_type, data_type)
            else:
                label_path = join(
                    '/n/sd8/inaguma/corpus/csj/dataset/labels/attention/',
                    train_data_size, label_type, data_type)

        # Load the frame number dictionary
        with open(join(input_path, 'frame_num.pickle'), 'rb') as f:
            self.frame_num_dict = pickle.load(f)

        # Sort paths to input & label by frame num
        print('=> Loading paths to dataset...')
        frame_num_tuple_sorted = sorted(self.frame_num_dict.items(),
                                        key=lambda x: x[1])
        input_paths, label_paths = [], []
        for input_name, frame_num in wrap_iterator(frame_num_tuple_sorted,
                                                   self.progressbar):
            speaker_name = input_name.split('_')[0]
            input_paths.append(
                join(input_path, speaker_name, input_name + '.npy'))
            label_paths.append(
                join(label_path, speaker_name, input_name + '.npy'))
        self.input_paths = np.array(input_paths)
        self.label_paths = np.array(label_paths)
        self.data_num = len(self.input_paths)

        self.rest = set(range(0, self.data_num, 1))

        if data_type in ['eval1', 'eval2', 'eval3'] and label_type != 'phone':
            self.is_test = True
        else:
            self.is_test = False
