#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import sys
import unittest
import tensorflow as tf

sys.path.append('../../')
from csj.data.load_dataset_multitask_ctc import Dataset
from utils.labels.character import num2char
from utils.labels.phone import num2phone
from utils.sparsetensor import sparsetensor2list


class TestReadDatasetCTC(unittest.TestCase):

    def test(self):
        self.check_loading(label_type_main='kanji',
                           label_type_second='kana',
                           num_gpu=1, is_sorted=True)
        self.check_loading(label_type_main='kanji',
                           label_type_second='phone',
                           num_gpu=1, is_sorted=True)
        self.check_loading(label_type_main='character',
                           label_type_second='phone',
                           num_gpu=1, is_sorted=True)
        self.check_loading(label_type_main='kanji',
                           label_type_second='kana',
                           num_gpu=1, is_sorted=False)
        self.check_loading(label_type_main='kanji',
                           label_type_second='phone',
                           num_gpu=1, is_sorted=False)
        self.check_loading(label_type_main='character',
                           label_type_second='phone',
                           num_gpu=1, is_sorted=False)

        # self.check_loading(label_type_main='kanji',
        #                    label_type_second='kana',
        #                    num_gpu=2, is_sorted=True)
        # self.check_loading(label_type_main='kanji',
        #                    label_type_second='phone',
        #                    num_gpu=2, is_sorted=True)
        # self.check_loading(label_type_main='character',
        #                    label_type_second='phone',
        #                    num_gpu=2, is_sorted=True)
        # self.check_loading(label_type_main='kanji',
        #                    label_type_second='kana',
        #                    num_gpu=2, is_sorted=False)
        # self.check_loading(label_type_main='kanji',
        #                    label_type_second='phone',
        #                    num_gpu=2, is_sorted=False)
        # self.check_loading(label_type_main='character',
        #                    label_type_second='phone',
        #                    num_gpu=2, is_sorted=False)

        # For many GPUs
        # self.check_loading(label_type_main='kanji',
        #                    label_type_second='kana',
        #                    num_gpu=7, is_sorted=True)

    def check_loading(self, label_type_main, label_type_second, num_gpu,
                      is_sorted):

        print('----- num_gpu: ' + str(num_gpu) +
              ', is_sorted: ' + str(is_sorted) + ' -----')

        batch_size = 64
        dataset = Dataset(data_type='dev', train_data_size='default',
                          label_type_main=label_type_main,
                          label_type_second=label_type_second,
                          batch_size=batch_size,
                          num_stack=3, num_skip=3,
                          is_sorted=is_sorted, is_progressbar=True,
                          num_gpu=num_gpu)

        tf.reset_default_graph()
        with tf.Session().as_default() as sess:
            print('=> Reading mini-batch...')
            if label_type_main == 'kanji':
                map_file_path_main = '../metrics/mapping_files/ctc/kanji2num.txt'
                map_fn_main = num2char
            elif label_type_main == 'kana':
                map_file_path_main = '../metrics/mapping_files/ctc/kana2num.txt'
                map_fn_main = num2char

            if label_type_second == 'kana':
                map_file_path_second = '../metrics/mapping_files/ctc/kana2num.txt'
                map_fn_second = num2char
            elif label_type_second == 'phone':
                map_file_path_second = '../metrics/mapping_files/ctc/phone2num.txt'
                map_fn_second = num2phone

            mini_batch = dataset.next_batch(session=sess)

            iter_per_epoch = int(dataset.data_num /
                                 (batch_size * num_gpu)) + 1
            for i in range(iter_per_epoch + 1):
                inputs, labels_main_st, labels_second_st, inputs_seq_len, input_names = mini_batch.__next__()

                if num_gpu > 1:
                    for inputs_gpu in inputs:
                        print(inputs_gpu.shape)
                    labels_main_st = labels_main_st[0]
                    labels_second_st = labels_second_st[0]

                labels_main = sparsetensor2list(
                    labels_main_st, batch_size=len(inputs))
                labels_second = sparsetensor2list(
                    labels_second_st, batch_size=len(inputs))

                if num_gpu < 1:
                    for inputs_i, labels_i in zip(inputs, labels_main):
                        if len(inputs_i) < len(labels_i):
                            print(len(inputs_i))
                            print(len(labels_i))
                            raise ValueError
                    for inputs_i, labels_i in zip(inputs, labels_second):
                        if len(inputs_i) < len(labels_i):
                            print(len(inputs_i))
                            print(len(labels_i))
                            raise ValueError

                str_true_main = map_fn_main(labels_main[0], map_file_path_main)
                str_true_main = re.sub(r'_', ' ', str_true_main)
                str_true_second = map_fn_second(
                    labels_second[0], map_file_path_second)
                print(str_true_main)
                print(str_true_second)
                print('-----')


if __name__ == '__main__':
    unittest.main()
