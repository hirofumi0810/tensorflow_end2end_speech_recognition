#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os
import sys
import unittest

sys.path.append(os.path.abspath('../../../../'))
from experiments.csj.data.load_dataset_multitask_ctc import Dataset
from utils.io.labels.character import idx2char
from utils.io.labels.phone import idx2phone
from utils.measure_time_func import measure_time


class TestReadDatasetCTC(unittest.TestCase):

    def test(self):

        # data_type
        self.check_loading(label_type_main='kanji', label_type_sub='kana',
                           data_type='train')
        self.check_loading(label_type_main='kanji', label_type_sub='kana',
                           data_type='dev')
        self.check_loading(label_type_main='kanji', label_type_sub='kana',
                           data_type='eval1')
        self.check_loading(label_type_main='kanji', label_type_sub='kana',
                           data_type='eval2')
        self.check_loading(label_type_main='kanji', label_type_sub='kana',
                           data_type='eval3')

        # label_type
        self.check_loading(label_type_main='kanji', label_type_sub='phone')
        self.check_loading(label_type_main='kanji_wakachi',
                           label_type_sub='phone')
        self.check_loading(label_type_main='kana', label_type_sub='phone')
        self.check_loading(label_type_main='kana_wakachi',
                           label_type_sub='phone')

        # sort
        self.check_loading(label_type_main='kanji', label_type_sub='kana',
                           sort_utt=True)
        self.check_loading(label_type_main='kanji', label_type_sub='kana',
                           sort_utt=True, sort_stop_epoch=True)

        # frame stacking
        self.check_loading(label_type_main='kanji', label_type_sub='kana',
                           frame_stacking=True)

        # splicing
        self.check_loading(label_type_main='kanji', label_type_sub='kana',
                           splice=11)

        # multi-GPU
        self.check_loading(label_type_main='kanji', label_type_sub='kana',
                           num_gpu=8)

    @measure_time
    def check_loading(self, label_type_main, label_type_sub, data_type='dev',
                      sort_utt=False, sort_stop_epoch=None,
                      frame_stacking=False, splice=1, num_gpu=1):

        print('========================================')
        print('  label_type_main: %s' % label_type_main)
        print('  label_type_sub: %s' % label_type_sub)
        print('  data_type: %s' % data_type)
        print('  sort_utt: %s' % str(sort_utt))
        print('  sort_stop_epoch: %s' % str(sort_stop_epoch))
        print('  frame_stacking: %s' % str(frame_stacking))
        print('  splice: %d' % splice)
        print('  num_gpu: %d' % num_gpu)
        print('========================================')

        num_stack = 3 if frame_stacking else 1
        num_skip = 3 if frame_stacking else 1
        dataset = Dataset(
            data_type=data_type, train_data_size='train_fullset',
            label_type_main=label_type_main,
            label_type_sub=label_type_sub,
            batch_size=64, max_epoch=2, splice=splice,
            num_stack=num_stack, num_skip=num_skip,
            sort_utt=sort_utt, sort_stop_epoch=sort_stop_epoch,
            progressbar=True, num_gpu=num_gpu)

        print('=> Loading mini-batch...')
        if label_type_main == 'kanji':
            map_file_path_main = '../../metrics/mapping_files/ctc/kanji.txt'
            map_fn_main = idx2char
        elif label_type_main == 'kana':
            map_file_path_main = '../../metrics/mapping_files/ctc/kana.txt'
            map_fn_main = idx2char

        if label_type_sub == 'kana':
            map_file_path_sub = '../../metrics/mapping_files/ctc/kana.txt'
            map_fn_sub = idx2char
        elif label_type_sub == 'phone':
            map_file_path_sub = '../../metrics/mapping_files/ctc/phone.txt'
            map_fn_sub = idx2phone

        for data, is_new_epoch in dataset:
            inputs, labels_main, labels_sub, inputs_seq_len, input_names = data

            if num_gpu > 1:
                for inputs_gpu in inputs:
                    print(inputs_gpu.shape)
                inputs = inputs[0]
                labels_main = labels_main[0]
                labels_sub = labels_sub[0]

            str_true_main = map_fn_main(labels_main[0], map_file_path_main)
            str_true_main = re.sub(r'_', ' ', str_true_main)
            str_true_sub = map_fn_sub(labels_sub[0], map_file_path_sub)
            print('----- %s (epoch: %.3f) -----' %
                  (input_names[0][0], dataset.epoch_detail))
            print(str_true_main)
            print(str_true_sub)


if __name__ == '__main__':
    unittest.main()
