#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.abspath('../../../../'))
from experiments.librispeech.data.load_dataset_ctc import Dataset
from utils.io.labels.character import Idx2char
from utils.io.labels.word import Idx2word
from utils.measure_time_func import measure_time


class TestLoadDatasetCTC(unittest.TestCase):

    def test(self):

        # data_type
        self.check(label_type='character', data_type='train')
        self.check(label_type='character', data_type='dev_clean')
        self.check(label_type='character', data_type='dev_other')
        self.check(label_type='character', data_type='test_clean')
        self.check(label_type='character', data_type='test_other')
        self.check(label_type='character_capital_divide', data_type='train')
        self.check(label_type='character_capital_divide',
                   data_type='dev_clean')
        self.check(label_type='character_capital_divide',
                   data_type='dev_other')
        self.check(label_type='character_capital_divide',
                   data_type='test_clean')
        self.check(label_type='character_capital_divide',
                   data_type='test_other')
        self.check(label_type='word_freq10', data_type='train')
        self.check(label_type='word_freq10', data_type='dev_clean')
        self.check(label_type='word_freq10', data_type='dev_other')
        self.check(label_type='word_freq10', data_type='test_clean')
        self.check(label_type='word_freq10', data_type='test_other')

        # sort
        self.check(label_type='character', sort_utt=True)
        self.check(label_type='character', sort_utt=True,
                   sort_stop_epoch=1)
        self.check(label_type='character', shuffle=True)

        # frame stacking
        self.check(label_type='character', frame_stacking=True)

        # splicing
        self.check(label_type='character', splice=11)

        # multi-GPU
        self.check(label_type='character', num_gpu=8)

    @measure_time
    def check(self, label_type, data_type='dev_clean',
              shuffle=False,  sort_utt=False, sort_stop_epoch=None,
              frame_stacking=False, splice=1, num_gpu=1):

        print('========================================')
        print('  label_type: %s' % label_type)
        print('  data_type: %s' % data_type)
        print('  shuffle: %s' % str(shuffle))
        print('  sort_utt: %s' % str(sort_utt))
        print('  sort_stop_epoch: %s' % str(sort_stop_epoch))
        print('  frame_stacking: %s' % str(frame_stacking))
        print('  splice: %d' % splice)
        print('  num_gpu: %d' % num_gpu)
        print('========================================')

        num_stack = 3 if frame_stacking else 1
        num_skip = 3 if frame_stacking else 1
        dataset = Dataset(
            data_type=data_type, train_data_size='train100h',
            label_type=label_type,
            batch_size=64, max_epoch=2, splice=splice,
            num_stack=num_stack, num_skip=num_skip,
            shuffle=shuffle, sort_utt=sort_utt, sort_stop_epoch=sort_stop_epoch,
            progressbar=True, num_gpu=num_gpu)

        print('=> Loading mini-batch...')
        if label_type == 'character':
            map_file_path = '../../metrics/mapping_files/character.txt'
        else:
            map_file_path = '../../metrics/mapping_files/' + label_type + '_' + \
                dataset.train_data_size + '.txt'

        idx2char = Idx2char(map_file_path)
        idx2word = Idx2word(map_file_path)

        for data, is_new_epoch in dataset:
            inputs, labels, inputs_seq_len, input_names = data

            if data_type == 'train':
                for i, l in zip(inputs[0], labels[0]):
                    if len(i) < len(l):
                        raise ValueError(
                            'input length must be longer than label length.')

            if num_gpu > 1:
                for inputs_gpu in inputs:
                    print(inputs_gpu.shape)

            if 'test' in data_type:
                str_true = labels[0][0][0]
            else:
                if 'word' in label_type:
                    str_true = '_'.join(idx2word(labels[0][0]))
                else:
                    str_true = idx2char(labels[0][0])

            print('----- %s (epoch: %.3f) -----' %
                  (input_names[0][0], dataset.epoch_detail))
            print(inputs[0].shape)
            print(str_true)

            if dataset.epoch_detail >= 0.1:
                break


if __name__ == '__main__':
    unittest.main()
