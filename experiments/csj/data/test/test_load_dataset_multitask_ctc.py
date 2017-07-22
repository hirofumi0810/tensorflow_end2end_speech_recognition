#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import sys
import unittest
import tensorflow as tf

sys.path.append('../../../../')
from experiments.csj.data.load_dataset_multitask_ctc import Dataset
from experiments.utils.labels.character import num2char
from experiments.utils.labels.phone import num2phone
from experiments.utils.measure_time_func import measure_time


class TestReadDatasetCTC(unittest.TestCase):

    def test(self):
        self.check_loading(label_type_main='kanji', label_type_sub='kana',
                           num_gpu=1, sort_utt=True)
        self.check_loading(label_type_main='kanji', label_type_sub='phone',
                           num_gpu=1, sort_utt=True)
        self.check_loading(label_type_main='kana', label_type_sub='phone',
                           num_gpu=1, sort_utt=True)
        self.check_loading(label_type_main='kanji', label_type_sub='kana',
                           num_gpu=1, sort_utt=False)
        self.check_loading(label_type_main='kanji', label_type_sub='phone',
                           num_gpu=1, sort_utt=False)
        self.check_loading(label_type_main='kana', label_type_sub='phone',
                           num_gpu=1, sort_utt=False)

        self.check_loading(label_type_main='kanji', label_type_sub='kana',
                           num_gpu=2, sort_utt=True)
        self.check_loading(label_type_main='kanji', label_type_sub='phone',
                           num_gpu=2, sort_utt=True)
        self.check_loading(label_type_main='kana', label_type_sub='phone',
                           num_gpu=2, sort_utt=True)
        self.check_loading(label_type_main='kanji', label_type_sub='kana',
                           num_gpu=2, sort_utt=False)
        self.check_loading(label_type_main='kanji', label_type_sub='phone',
                           num_gpu=2, sort_utt=False)
        self.check_loading(label_type_main='kana', label_type_sub='phone',
                           num_gpu=2, sort_utt=False)

        # For many GPUs
        self.check_loading(label_type_main='kanji', label_type_sub='kana',
                           num_gpu=7, sort_utt=True)

    @measure_time
    def check_loading(self, label_type_main, label_type_sub, num_gpu,
                      sort_utt):

        print('----- num_gpu: ' + str(num_gpu) +
              ', sort_utt: ' + str(sort_utt) + ' -----')

        batch_size = 64
        dataset = Dataset(data_type='dev', train_data_size='default',
                          label_type_main=label_type_main,
                          label_type_sub=label_type_sub,
                          batch_size=batch_size,
                          num_stack=3, num_skip=3,
                          sort_utt=sort_utt, progressbar=True,
                          num_gpu=num_gpu)

        tf.reset_default_graph()
        with tf.Session().as_default() as sess:
            print('=> Loading mini-batch...')
            if label_type_main == 'kanji':
                map_file_path_main = '../../metrics/mapping_files/ctc/kanji2num.txt'
                map_fn_main = num2char
            elif label_type_main == 'kana':
                map_file_path_main = '../../metrics/mapping_files/ctc/kana2num.txt'
                map_fn_main = num2char

            if label_type_sub == 'kana':
                map_file_path_sub = '../../metrics/mapping_files/ctc/kana2num.txt'
                map_fn_sub = num2char
            elif label_type_sub == 'phone':
                map_file_path_sub = '../../metrics/mapping_files/ctc/phone2num.txt'
                map_fn_sub = num2phone

            mini_batch = dataset.next_batch(session=sess)

            iter_per_epoch = int(dataset.data_num /
                                 (batch_size * num_gpu)) + 1
            for i in range(iter_per_epoch + 1):
                inputs, labels_main, labels_sub, inputs_seq_len, input_names = mini_batch.__next__()

                if num_gpu > 1:
                    for inputs_gpu in inputs:
                        print(inputs_gpu.shape)
                    labels_main = labels_main[0]
                    labels_sub = labels_sub[0]

                if num_gpu == 1:
                    for inputs_i, labels_i in zip(inputs, labels_main):
                        if len(inputs_i) < len(labels_i):
                            print(len(inputs_i))
                            print(len(labels_i))
                            raise ValueError
                    for inputs_i, labels_i in zip(inputs, labels_sub):
                        if len(inputs_i) < len(labels_i):
                            print(len(inputs_i))
                            print(len(labels_i))
                            raise ValueError

                str_true_main = map_fn_main(labels_main[0], map_file_path_main)
                str_true_main = re.sub(r'_', ' ', str_true_main)
                str_true_sub = map_fn_sub(labels_sub[0], map_file_path_sub)
                print(str_true_main)
                print(str_true_sub)
                print('-----')


if __name__ == '__main__':
    unittest.main()
