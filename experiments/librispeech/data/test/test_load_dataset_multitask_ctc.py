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
from experiments.librispeech.data.load_dataset_multitask_ctc import Dataset
from utils.io.labels.character import Idx2char
from utils.io.labels.word import Idx2word
from utils.measure_time_func import measure_time


class TestLoadDatasetMultitaskCTC(unittest.TestCase):

    def test(self):

        # data_type
        self.check(label_type_sub='character', data_type='train')
        self.check(label_type_sub='character', data_type='dev_clean')
        self.check(label_type_sub='character', data_type='dev_other')
        self.check(label_type_sub='character', data_type='test_clean')
        self.check(label_type_sub='character', data_type='test_other')
        self.check(label_type_sub='character_capital_divide',
                   data_type='train')
        self.check(label_type_sub='character_capital_divide',
                   data_type='dev_clean')
        self.check(label_type_sub='character_capital_divide',
                   data_type='dev_other')
        self.check(label_type_sub='character_capital_divide',
                   data_type='test_clean')
        self.check(label_type_sub='character_capital_divide',
                   data_type='test_other')

        # sort
        self.check(label_type_sub='character', sort_utt=True)
        self.check(label_type_sub='character', sort_utt=True,
                   sort_stop_epoch=2)
        self.check(label_type_sub='character', shuffle=True)

        # frame stacking
        self.check(label_type_sub='character', frame_stacking=True)

        # splicing
        self.check(label_type_sub='character', splice=11)

        # multi-GPU
        self.check(label_type_sub='character', num_gpu=8)

    @measure_time
    def check(self, label_type_sub, data_type='dev_clean',
              shuffle=False,  sort_utt=False, sort_stop_epoch=None,
              frame_stacking=False, splice=1, num_gpu=1):

        print('========================================')
        print('  label_type_main: %s' % 'word')
        print('  label_type_sub: %s' % label_type_sub)
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
            label_type_main='word_freq10', label_type_sub=label_type_sub,
            batch_size=64, max_epoch=1, splice=splice,
            num_stack=num_stack, num_skip=num_skip,
            shuffle=shuffle, sort_utt=sort_utt, sort_stop_epoch=sort_stop_epoch,
            progressbar=True, num_gpu=num_gpu)

        print('=> Loading mini-batch...')
        if label_type_sub == 'character':
            map_file_path_char = '../../metrics/mapping_files/character.txt'
        elif label_type_sub == 'character_capital_divide':
            map_file_path_char = '../../metrics/mapping_files/character_capital.txt'
        map_file_path_word = '../../metrics/mapping_files/word_' + \
            dataset.train_data_size + '.txt'

        idx2char = Idx2char(map_file_path=map_file_path_char)
        idx2word = Idx2word(map_file_path=map_file_path_word)

        for data, is_new_epoch in dataset:
            inputs, labels_word, labels_char, inputs_seq_len, input_names = data

            if data_type == 'train':
                for i, l in zip(inputs[0], labels_char[0]):
                    if len(i) < len(l):
                        raise ValueError(
                            'input length must be longer than label length.')

            if num_gpu > 1:
                for inputs_gpu in inputs:
                    print(inputs_gpu.shape)

            if 'test' not in data_type:
                word_list = idx2word(labels_word[0][0])
                str_true_char = idx2char(labels_char[0][0],
                                         padded_value=dataset.padded_value)
                str_true_char = re.sub(r'_', ' ', str_true_char)
            else:
                word_list = np.delete(labels_word[0][0], np.where(
                    labels_word[0][0] == None), axis=0)
                str_true_char = labels_char[0][0][0]
            str_true_word = ' '.join(word_list)

            print('----- %s (epoch: %.3f) -----' %
                  (input_names[0][0], dataset.epoch_detail))
            print(inputs[0].shape)
            print(str_true_word)
            print(str_true_char)

            if dataset.epoch_detail >= 0.1:
                break


if __name__ == '__main__':
    unittest.main()
