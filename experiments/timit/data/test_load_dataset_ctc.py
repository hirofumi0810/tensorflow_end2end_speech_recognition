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
from experiments.timit.data.load_dataset_ctc import Dataset
from experiments.utils.labels.character import num2char
from experiments.utils.labels.phone import num2phone
from experiments.utils.sparsetensor import sparsetensor2list
from experiments.utils.measure_time_func import measure_time


class TestLoadDatasetCTC(unittest.TestCase):

    def test(self):
        self.check_loading(label_type='character', num_gpu=1, is_sorted=True)
        self.check_loading(label_type='character', num_gpu=1, is_sorted=False)
        self.check_loading(label_type='phone61', num_gpu=1, is_sorted=True)
        self.check_loading(label_type='phone61', num_gpu=1, is_sorted=False)

        # self.check_loading(label_type='character', num_gpu=2, is_sorted=True)
        # self.check_loading(label_type='character', num_gpu=2, is_sorted=False)
        # self.check_loading(label_type='phone61', num_gpu=2, is_sorted=True)
        # self.check_loading(label_type='phone61', num_gpu=2, is_sorted=False)

        # For many GPUs
        # self.check_loading(label_type='character', num_gpu=7, is_sorted=True)

    @measure_time
    def check_loading(self, label_type, num_gpu, is_sorted):
        print('----- label_type: ' + label_type + ', num_gpu: ' +
              str(num_gpu) + ', is_sorted: ' + str(is_sorted) + ' -----')

        batch_size = 64
        dataset = Dataset(data_type='train', label_type=label_type,
                          batch_size=batch_size,
                          num_stack=3, num_skip=3,
                          is_sorted=is_sorted, is_progressbar=True,
                          num_gpu=num_gpu)

        tf.reset_default_graph()
        with tf.Session().as_default() as sess:
            print('=> Loading mini-batch...')
            if label_type == 'character':
                map_file_path = '../metrics/mapping_files/ctc/char2num.txt'
                map_fn = num2char
            else:
                map_file_path = '../metrics/mapping_files/ctc/phone2num_' + \
                    label_type[5:7] + '.txt'
                map_fn = num2phone

            mini_batch = dataset.next_batch(session=sess)

            iter_per_epoch = int(dataset.data_num /
                                 (batch_size * num_gpu)) + 1
            for i in range(iter_per_epoch + 1):
                return_tuple = mini_batch.__next__()
                inputs = return_tuple[0]
                labels_st = return_tuple[1]

                if num_gpu > 1:
                    for inputs_gpu in inputs:
                        print(inputs_gpu.shape)
                    labels_st = labels_st[0]

                labels = sparsetensor2list(
                    labels_st, batch_size=len(inputs))

                if num_gpu == 1:
                    for inputs_i, labels_i in zip(inputs, labels):
                        if len(inputs_i) < len(labels_i):
                            print(len(inputs_i))
                            print(len(labels_i))
                            raise ValueError

                str_true = map_fn(labels[0], map_file_path)
                str_true = re.sub(r'_', ' ', str_true)
                # print(str_true)


if __name__ == '__main__':
    unittest.main()
