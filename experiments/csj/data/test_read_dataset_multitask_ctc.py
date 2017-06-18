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


class TestReadDatasetCTC(unittest.TestCase):

    def test(self):
        self.check_reading(label_type_main='kanji', label_type_second='phone',
                           num_gpu=1, is_sorted=True)

    def check_reading(self, label_type_main, label_type_second, num_gpu, is_sorted):

        print('----- num_gpu: ' + str(num_gpu) +
              ', is_sorted: ' + str(is_sorted) + ' -----')
        dataset = DataSet(data_type='eval1', train_data_size='default',
                          label_type_main=label_type_main,
                          label_type_second=label_type_second,
                          batch_size=64,
                          num_stack=3, num_skip=3,
                          is_sorted=True, is_progressbar=True,
                          num_gpu=num_gpu)

        with tf.Session().as_default() as sess:
            print('=> Reading mini-batch...')
            map_file_path_main = '../metric/mapping_files/ctc/char2num.txt'
            map_file_path_second = '../metric/mapping_files/ctc/phone2num_61.txt'

            for i in tqdm(range(10)):
                inputs, labels_main_st, labels_second_st, inputs_seq_len, input_names = dataset.next_batch(
                    session=sess)

                if num_gpu > 1:
                    labels_main_st = labels_main_st[0]
                    labels_second_st = labels_second_st[0]

                labels_main = sparsetensor2list(
                    labels_main_st, batch_size=len(labels_main_st))
                labels_second = sparsetensor2list(
                    labels_second_st, batch_size=len(labels_second_st))
                str_true_main = num2char(labels_main[0], map_file_path_main)
                # str_true_main = re.sub(r'_', ' ', str_true_main)
                str_true_second = num2phone(
                    labels_second[0], map_file_path_second)
                print(str_true_main)
                print(str_true_second)
                print('-----')


if __name__ == '__main__':
    unittest.main()
