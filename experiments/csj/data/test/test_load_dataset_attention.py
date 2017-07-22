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
from experiments.csj.data.load_dataset_attention import Dataset
from experiments.utils.labels.character import num2char
from experiments.utils.labels.phone import num2phone
from experiments.utils.measure_time_func import measure_time


class TestLoadDatasetAttention(unittest.TestCase):
    def test(self):
        self.check_loading(label_type='kanji', num_gpu=1, sort_utt=True)
        self.check_loading(label_type='kana', num_gpu=1, sort_utt=True)
        self.check_loading(label_type='phone', num_gpu=1, sort_utt=True)

        self.check_loading(label_type='kanji', num_gpu=1, sort_utt=False)
        self.check_loading(label_type='kana', num_gpu=1, sort_utt=False)
        self.check_loading(label_type='phone', num_gpu=1, sort_utt=False)

        self.check_loading(label_type='kanji', num_gpu=2, sort_utt=True)
        self.check_loading(label_type='kana', num_gpu=2, sort_utt=True)
        self.check_loading(label_type='phone', num_gpu=2, sort_utt=True)

        self.check_loading(label_type='kanji', num_gpu=2, sort_utt=False)
        self.check_loading(label_type='kana', num_gpu=2, sort_utt=False)
        self.check_loading(label_type='phone', num_gpu=2, sort_utt=False)

        # For many GPUs
        self.check_loading(label_type='kanji', num_gpu=7, sort_utt=True)

    @measure_time
    def check_loading(self, label_type, num_gpu, sort_utt):
        print('----- label_type: ' + label_type + ', num_gpu: ' +
              str(num_gpu) + ', sort_utt: ' + str(sort_utt) + ' -----')

        batch_size = 64
        dataset = Dataset(data_type='dev', train_data_size='default',
                          label_type=label_type, batch_size=batch_size,
                          eos_index=1,
                          sort_utt=sort_utt, progressbar=True,
                          num_gpu=num_gpu)

        tf.reset_default_graph()
        with tf.Session().as_default() as sess:
            print('=> Loading mini-batch...')
            if label_type == 'kanji':
                map_file_path = '../../metrics/mapping_files/attention/kanji2num.txt'
                map_fn = num2char
            elif label_type == 'kana':
                map_file_path = '../../metrics/mapping_files/attention/kana2num.txt'
                map_fn = num2char
            elif label_type == 'phone':
                map_file_path = '../../metrics/mapping_files/attention/phone2num.txt'
                map_fn = num2phone

            mini_batch = dataset.next_batch(session=sess)

            iter_per_epoch = int(dataset.data_num /
                                 (batch_size * num_gpu)) + 1
            for i in range(iter_per_epoch + 1):
                inputs, labels, inputs_seq_len, labels_seq_len, _ = mini_batch.__next__()

                if num_gpu > 1:
                    for inputs_gpu in inputs:
                        print(inputs_gpu.shape)
                    inputs = inputs[0]
                    labels = labels[0]
                    labels_seq_len = labels_seq_len[0]

                str_true = map_fn(
                    labels[0][0: labels_seq_len[0]], map_file_path)
                str_true = re.sub(r'_', ' ', str_true)
                print(str_true)


if __name__ == '__main__':
    unittest.main()
