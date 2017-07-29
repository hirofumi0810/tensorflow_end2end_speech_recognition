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
from experiments.timit.data.load_dataset_multitask_ctc import Dataset
from experiments.utils.data.labels.character import num2char
from experiments.utils.data.labels.phone import num2phone
from experiments.utils.measure_time_func import measure_time


class TestLoadDatasetMultitaskCTC(unittest.TestCase):

    def test(self):
        # label_type
        self.check_loading(label_type_main='character',
                           num_gpu=1, sort_utt=False, sorta_grad=False)
        self.check_loading(label_type_main='character_capital_divide',
                           num_gpu=1, sort_utt=False, sorta_grad=False)

        # sort
        self.check_loading(label_type_main='character',
                           num_gpu=1, sort_utt=True, sorta_grad=False)
        self.check_loading(label_type_main='character',
                           num_gpu=1, sort_utt=False, sorta_grad=True)

        # multi-GPU
        self.check_loading(label_type_main='character',
                           num_gpu=2, sort_utt=False, sorta_grad=False)
        self.check_loading(label_type_main='character',
                           num_gpu=7, sort_utt=False, sorta_grad=False)

    @measure_time
    def check_loading(self, label_type_main, num_gpu, sort_utt, sorta_grad):
        print('----- label_type_main: %s, um_gpu: %d, sort_utt: %s, sorta_grad: %s -----' %
              (label_type_main, num_gpu, str(sort_utt), str(sorta_grad)))

        dataset = Dataset(data_type='dev',
                          label_type_main=label_type_main,
                          label_type_sub='phone61',
                          batch_size=64,
                          num_stack=3, num_skip=3,
                          sort_utt=sort_utt, sorta_grad=sorta_grad,
                          progressbar=True, num_gpu=num_gpu)

        tf.reset_default_graph()
        with tf.Session().as_default() as sess:
            print('=> Loading mini-batch...')
            if label_type_main == 'character_capital_divide':
                map_file_path_char = '../../metrics/mapping_files/ctc/character_to_num_capital.txt'
            else:
                map_file_path_char = '../../metrics/mapping_files/ctc/character_to_num.txt'
            map_file_path_phone = '../../metrics/mapping_files/ctc/phone61_to_num.txt'

            for data, next_epoch_flag in dataset(session=sess):
                inputs, labels_char, labels_phone, _, _ = data

                if num_gpu > 1:
                    # for inputs_gpu in inputs:
                    #     print(inputs_gpu.shape)
                    labels_char = labels_char[0]
                    labels_phone = labels_phone[0]

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
                print(str_true_char)
                print(str_true_phone)
                print('-----')

                if next_epoch_flag:
                    break


if __name__ == '__main__':
    unittest.main()
