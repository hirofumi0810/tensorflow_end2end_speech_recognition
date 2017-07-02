#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import sys
import unittest
import tensorflow as tf

sys.path.append('../../../')
from experiments.timit.data.load_dataset_multitask_ctc import Dataset
from experiments.utils.labels.character import num2char
from experiments.utils.labels.phone import num2phone
from experiments.utils.sparsetensor import sparsetensor2list
from experiments.utils.measure_time_func import measure_time


class TestLoadDatasetMultitaskCTC(unittest.TestCase):

    def test(self):
        self.check_loading(num_gpu=1, is_sorted=True)
        self.check_loading(num_gpu=1, is_sorted=False)

        # self.check_loading(num_gpu=2, is_sorted=True)
        # self.check_loading(num_gpu=2, is_sorted=False)

        # For many GPUs
        # self.check_loading(num_gpu=7, is_sorted=True)

    @measure_time
    def check_loading(self, num_gpu, is_sorted):
        print('----- num_gpu: ' + str(num_gpu) +
              ', is_sorted: ' + str(is_sorted) + ' -----')

        batch_size = 64
        dataset = Dataset(data_type='train',
                          label_type_main='character',
                          label_type_sub='phone61',
                          batch_size=batch_size,
                          num_stack=3, num_skip=3,
                          is_sorted=is_sorted, is_progressbar=True,
                          num_gpu=num_gpu)

        tf.reset_default_graph()
        with tf.Session().as_default() as sess:
            print('=> Loading mini-batch...')
            map_file_path_char = '../metrics/mapping_files/ctc/char2num.txt'
            map_file_path_phone = '../metrics/mapping_files/ctc/phone2num_61.txt'

            mini_batch = dataset.next_batch(session=sess)

            iter_per_epoch = int(dataset.data_num /
                                 (batch_size * num_gpu)) + 1
            for i in range(iter_per_epoch + 1):
                return_tuple = mini_batch.__next__()
                inputs = return_tuple[0]
                labels_char_st = return_tuple[1]
                labels_phone_st = return_tuple[2]

                if num_gpu > 1:
                    for inputs_gpu in inputs:
                        print(inputs_gpu.shape)
                    labels_char_st = labels_char_st[0]
                    labels_phone_st = labels_phone_st[0]

                labels_char = sparsetensor2list(
                    labels_char_st, batch_size=len(inputs))
                labels_phone = sparsetensor2list(
                    labels_phone_st, batch_size=len(inputs))

                if num_gpu == 1:
                    for inputs_i, labels_i in zip(inputs, labels_char):
                        if len(inputs_i) < len(labels_i):
                            print(len(inputs_i))
                            print(len(labels_i))
                            raise ValueError
                    for inputs_i, labels_i in zip(inputs, labels_phone):
                        if len(inputs_i) < len(labels_i):
                            print(len(inputs_i))
                            print(len(labels_i))
                            raise ValueError

                str_true_char = num2char(labels_char[0], map_file_path_char)
                str_true_char = re.sub(r'_', ' ', str_true_char)
                str_true_phone = num2phone(
                    labels_phone[0], map_file_path_phone)
                # print(str_true_char)
                # print(str_true_phone)
                # print('-----')


if __name__ == '__main__':
    unittest.main()
