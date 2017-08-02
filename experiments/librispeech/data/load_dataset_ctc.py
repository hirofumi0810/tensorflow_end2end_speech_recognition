#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Load dataset for the CTC model (Librispeech corpus).
   In addition, frame stacking and skipping are used.
   You can use the multi-GPU version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, isfile
import pickle
import numpy as np

from experiments.utils.data.dataset_loader.each_load.ctc_each_load import DatasetBase


class Dataset(DatasetBase):

    def __init__(self, data_type, train_data_size, label_type, batch_size,
                 num_stack=None, num_skip=None,
                 sort_utt=True, sort_stop_epoch=None,
                 progressbar=False, num_gpu=1, is_gpu=False):
        """A class for loading dataset.
        Args:
            data_type: string, train_clean100 or train_clean360 or
                train_other500 or train_all or dev_clean or dev_other or
                test_clean or test_other
            train_data_size: string, train_clean100 or train_clean360 or
                train_other500 or train_all
            label_type: string, character or character_capital_divide or word
            batch_size: int, the size of mini-batch
            num_stack: int, the number of frames to stack
            num_skip: int, the number of frames to skip
            sort_utt: if True, sort all utterances by the number of frames and
                utteraces in each mini-batch are shuffled
            sort_stop_epoch: After sort_stop_epoch, training will revert back
                to a random order
            progressbar: if True, visualize progressbar
            num_gpu: int, if more than 1, divide batch_size by num_gpu
            is_gpu: bool, if True, use dataset in the GPU server. This is
                useful when data size is very large and you cannot load all
                dataset at once. Then, you should put dataset on the GPU server
                you will use to reduce data-communication time between servers.
        """
        if data_type not in ['train_clean100', 'train_clean360',
                             'train_other500', 'train_all',
                             'dev_clean', 'dev_other',
                             'test_clean', 'test_other']:
            raise ValueError(
                'data_type is "train_clean100" or "train_clean360" or ' +
                '"train_other500" or "train_all" or "dev_clean" or ' +
                '"dev_other" or "test_clean" "test_other".')
        if data_type in ['train_clean100', 'train_clean360',
                         'train_other500', 'train_all']:
            self.is_training = True
        else:
            self.is_training = False
        if data_type in ['test_clean', 'test_other'] and label_type == 'word':
            self.is_test = True
        else:
            self.is_test = False

        self.data_type = data_type
        self.train_data_size = train_data_size
        self.label_type = label_type
        self.batch_size = batch_size * num_gpu
        self.num_stack = num_stack
        self.num_skip = num_skip
        self.sort_utt = sort_utt
        self.sort_stop_epoch = sort_stop_epoch
        self.epoch = 0
        self.progressbar = progressbar
        self.num_gpu = num_gpu
        self.input_size = None

        if is_gpu:
            # GPU server
            if data_type == 'train_all':
                input_path_list = [
                    '/data/inaguma/librispeech/inputs/train_clean100/train_clean100',
                    '/data/inaguma/librispeech/inputs/train_clean360/train_clean360',
                    '/data/inaguma/librispeech/inputs/train_other500/train_other500']
                label_path_list = [
                    join('/data/inaguma/librispeech/labels/ctc/train_clean100',
                         label_type, 'train_clean100'),
                    join('/data/inaguma/librispeech/labels/ctc/train_clean360',
                         label_type, 'train_clean360'),
                    join('/data/inaguma/librispeech/labels/ctc/train_other500',
                         label_type, 'train_other500')]
            else:
                input_path_list = [
                    join('/data/inaguma/librispeech/inputs',
                         train_data_size, data_type)]
                label_path_list = [
                    join('/data/inaguma/librispeech/labels/ctc',
                         train_data_size, label_type, data_type)]
        else:
            # CPU
            if data_type == 'train_all':
                input_path_list = [
                    '/n/sd8/inaguma/corpus/librispeech/dataset/inputs/train_clean100/train_clean100',
                    '/n/sd8/inaguma/corpus/librispeech/dataset/inputs/train_clean360/train_clean360',
                    '/n/sd8/inaguma/corpus/librispeech/dataset/inputs/train_other500/train_other500']
                label_path_list = [
                    join('/n/sd8/inaguma/corpus/librispeech/dataset/labels/ctc/train_clean100',
                         label_type, 'train_clean100'),
                    join('/n/sd8/inaguma/corpus/librispeech/dataset/labels/ctc/train_clean360',
                         label_type, 'train_clean360'),
                    join('/n/sd8/inaguma/corpus/librispeech/dataset/labels/ctc/train_other500',
                         label_type, 'train_other500')]
            else:
                input_path_list = [
                    join('/n/sd8/inaguma/corpus/librispeech/dataset/inputs',
                         train_data_size, data_type)]
                label_path_list = [
                    join('/n/sd8/inaguma/corpus/librispeech/dataset/labels/ctc',
                         train_data_size, label_type, data_type)]

        # Load the frame number dictionary
        if data_type == 'train_all':
            self.frame_num_dict = {}
            for input_path in input_path_list:
                with open(join(input_path, 'frame_num.pickle'), 'rb') as f:
                    self.frame_num_dict.update(pickle.load(f))
        else:
            with open(join(input_path_list[0], 'frame_num.pickle'), 'rb') as f:
                self.frame_num_dict = pickle.load(f)

        # Sort paths to input & label by frame num
        frame_num_tuple_sorted = sorted(self.frame_num_dict.items(),
                                        key=lambda x: x[1])
        input_paths, label_paths = [], []
        for input_name, frame_num in frame_num_tuple_sorted:
            speaker_name = input_name.split('-')[0]
            for input_path in input_path_list:
                if isfile(join(input_path, speaker_name, input_name + '.npy')):
                    input_paths.append(
                        join(input_path, speaker_name, input_name + '.npy'))
                    break
            for label_path in label_path_list:
                if isfile(join(label_path, speaker_name, input_name + '.npy')):
                    label_paths.append(
                        join(label_path, speaker_name, input_name + '.npy'))
                    break
        self.input_paths = np.array(input_paths)
        self.label_paths = np.array(label_paths)
        self.data_num = len(self.input_paths)
        # NOTE: Not load dataset yet

        assert len(self.input_paths) == len(self.label_paths), "Inputs and labels must have the same number of files (inputs: {0}, labels: {1}).".format(
            len(self.input_paths), len(self.label_paths))

        self.rest = set(range(0, self.data_num, 1))
