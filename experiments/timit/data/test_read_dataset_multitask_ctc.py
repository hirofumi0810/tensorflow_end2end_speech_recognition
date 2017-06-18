#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import sys
import unittest
from tqdm import tqdm
import tensorflow as tf

sys.path.append('../../')
sys.path.append('../../../')
from read_dataset_multitask_ctc import DataSet
from utils.labels.character import num2char
from utils.labels.phone import num2phone
from utils.sparsetensor import sparsetensor2list


class TestReadDatasetMultitaskCTC(unittest.TestCase):

    def test(self):
        self.check_reading(num_gpu=1, is_sorted=True)
        self.check_reading(num_gpu=1, is_sorted=False)
        self.check_reading(num_gpu=2, is_sorted=True)
        self.check_reading(num_gpu=2, is_sorted=False)

    def check_reading(self, num_gpu, is_sorted):
        print('----- num_gpu: ' + str(num_gpu) +
              ', is_sorted: ' + str(is_sorted) + ' -----')
        dataset = DataSet(data_type='test', label_type_second='phone61',
                          batch_size=64,
                          num_stack=3, num_skip=3,
                          is_sorted=is_sorted, is_progressbar=True,
                          num_gpu=num_gpu)

        with tf.Session().as_default() as sess:
            print('=> Reading mini-batch...')
            map_file_path_char = '../metric/mapping_files/ctc/char2num.txt'
            map_file_path_phone = '../metric/mapping_files/ctc/phone2num_61.txt'

            for i in tqdm(range(10)):
                inputs, labels_char_st, labels_phone_st, inputs_seq_len, input_names = dataset.next_batch(
                    session=sess)

                if num_gpu > 1:
                    labels_char_st = labels_char_st[0]
                    labels_phone_st = labels_phone_st[0]

                labels_char = sparsetensor2list(
                    labels_char_st, batch_size=len(labels_char_st))
                labels_phone = sparsetensor2list(
                    labels_phone_st, batch_size=len(labels_phone_st))
                str_true_char = num2char(labels_char[0], map_file_path_char)
                str_true_char = re.sub(r'_', ' ', str_true_char)
                str_true_phone = num2phone(
                    labels_phone[0], map_file_path_phone)
                print(str_true_char)
                print(str_true_phone)
                print('-----')


if __name__ == '__main__':
    unittest.main()
