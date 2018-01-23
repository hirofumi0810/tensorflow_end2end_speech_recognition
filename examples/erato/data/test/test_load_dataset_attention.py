#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.abspath('../../../../'))
from experiments.erato.data.load_dataset_attention import Dataset
from utils.io.labels.character import Idx2char
from utils.measure_time_func import measure_time


class TestLoadDatasetAttention(unittest.TestCase):

    def test(self):

        # data_type
        self.check(ss_type='insert_both', data_type='train')
        self.check(ss_type='insert_both', data_type='dev')
        self.check(ss_type='insert_both', data_type='test')

        # ss_type
        self.check(ss_type='insert_left', data_type='dev')
        self.check(ss_type='insert_right', data_type='dev')
        self.check(ss_type='remove', data_type='dev')

        # sort
        self.check(ss_type='insert_both', sort_utt=True)
        self.check(ss_type='insert_both', sort_utt=True,
                   sort_stop_epoch=2)
        self.check(ss_type='insert_both', shuffle=True)

        # frame stacking
        self.check(ss_type='insert_both', frame_stacking=True)

        # splicing
        self.check(ss_type='insert_both', splice=11)

    @measure_time
    def check(self, ss_type, data_type='dev',
              shuffle=False, sort_utt=False, sort_stop_epoch=None,
              frame_stacking=False, splice=1):

        print('========================================')
        print('  label_type: %s' % 'kana')
        print('  ss_type: %s' % ss_type)
        print('  data_type: %s' % data_type)
        print('  shuffle: %s' % str(shuffle))
        print('  sort_utt: %s' % str(sort_utt))
        print('  sort_stop_epoch: %s' % str(sort_stop_epoch))
        print('  frame_stacking: %s' % str(frame_stacking))
        print('  splice: %d' % splice)
        print('========================================')

        map_file_path = '../../metrics/mapping_files/kana_' + ss_type + '.txt'

        num_stack = 3 if frame_stacking else 1
        num_skip = 3 if frame_stacking else 1
        dataset = Dataset(
            data_type=data_type, label_type='kana', ss_type=ss_type,
            batch_size=64, map_file_path=map_file_path,
            max_epoch=2, splice=splice,
            num_stack=num_stack, num_skip=num_skip,
            shuffle=shuffle,
            sort_utt=sort_utt, sort_stop_epoch=sort_stop_epoch,
            progressbar=True)

        print('=> Loading mini-batch...')
        map_fn = Idx2char(map_file_path=map_file_path)

        for data, is_new_epoch in dataset:
            inputs, labels, inputs_seq_len, labels_seq_len, input_names = data

            if data_type == 'train':
                for i_batch, l_batch in zip(inputs[0], labels[0]):
                    if len(np.where(l_batch == dataset.padded_value)[0]) > 0:
                        if i_batch.shape[0] < np.where(l_batch == dataset.padded_value)[0][0]:
                            raise ValueError(
                                'input length must be longer than label length.')
                    else:
                        if i_batch.shape[0] < len(l_batch):
                            raise ValueError(
                                'input length must be longer than label length.')

            if data_type != 'test':
                str_true = map_fn(labels[0][0][:labels_seq_len[0][0]])
            else:
                str_true = labels[0][0][0]

            print('----- %s ----- (epoch: %.3f)' %
                  (input_names[0][0], dataset.epoch_detail))
            print(inputs[0][0].shape)
            print(str_true)

            if dataset.epoch_detail >= 0.2:
                break


if __name__ == '__main__':
    unittest.main()
