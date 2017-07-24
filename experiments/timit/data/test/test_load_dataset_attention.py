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
from experiments.timit.data.load_dataset_attention import Dataset
from experiments.utils.data.labels.character import num2char
from experiments.utils.data.labels.phone import num2phone
from experiments.utils.measure_time_func import measure_time


class TestLoadDatasetAttention(unittest.TestCase):

    def test(self):

        # label_type
        self.check_loading(label_type='character', num_gpu=1, sort_utt=False)
        self.check_loading(label_type='phone61', num_gpu=1, sort_utt=False)

        # sort
        self.check_loading(label_type='phone61', num_gpu=1, sort_utt=True)

        # multi-GPU
        self.check_loading(label_type='phone61', num_gpu=2, sort_utt=True)
        self.check_loading(label_type='phone61', num_gpu=7, sort_utt=True)

    @measure_time
    def check_loading(self, label_type, num_gpu, sort_utt):
        print('----- label_type: ' + label_type + ', num_gpu: ' +
              str(num_gpu) + ', sort_utt: ' + str(sort_utt) + ' -----')

        batch_size = 64
        eos_index = 2 if label_type == 'character' else 1
        dataset = Dataset(data_type='train', label_type=label_type,
                          batch_size=batch_size, eos_index=eos_index,
                          sort_utt=sort_utt, progressbar=True,
                          num_gpu=num_gpu)

        tf.reset_default_graph()
        with tf.Session().as_default() as sess:
            print('=> Loading mini-batch...')
            map_file_path = '../../metrics/mapping_files/attention/' + label_type + '_to_num.txt'
            if label_type == 'character':
                map_fn = num2char
            else:
                map_fn = num2phone

            mini_batch = dataset.next_batch(session=sess)

            iter_per_epoch = int(dataset.data_num /
                                 (batch_size * num_gpu)) + 1
            for i in range(iter_per_epoch + 1):
                inputs, labels, inputs_seq_len, labels_seq_len, _ = mini_batch.__next__()

                if num_gpu > 1:
                    # for inputs_gpu in inputs:
                    #     print(inputs_gpu.shape)
                    inputs = inputs[0]
                    labels = labels[0]
                    labels_seq_len = labels_seq_len[0]

                str_true = map_fn(
                    labels[0][0: labels_seq_len[0]], map_file_path)
                str_true = re.sub(r'_', ' ', str_true)
                print(str_true)


if __name__ == '__main__':
    unittest.main()
