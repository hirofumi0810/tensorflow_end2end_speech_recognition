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
        self.check_loading(label_type='character', num_gpu=1,
                           sort_utt=False, sorta_grad=False)
        self.check_loading(label_type='character_capital_divide', num_gpu=1,
                           sort_utt=False, sorta_grad=False)
        self.check_loading(label_type='phone61', num_gpu=1,
                           sort_utt=False, sorta_grad=False)

        # sort
        self.check_loading(label_type='phone61', num_gpu=1,
                           sort_utt=True, sorta_grad=False)
        self.check_loading(label_type='phone61', num_gpu=1,
                           sort_utt=False, sorta_grad=True)

        # multi-GPU
        self.check_loading(label_type='phone61', num_gpu=2,
                           sort_utt=False, sorta_grad=False)
        self.check_loading(label_type='phone61', num_gpu=7,
                           sort_utt=False, sorta_grad=False)

    @measure_time
    def check_loading(self, label_type, num_gpu, sort_utt, sorta_grad):
        print('----- label_type: %s, num_gpu: %d, sort_utt: %s, sorta_grad: %s -----' %
              (label_type, num_gpu, str(sort_utt), str(sorta_grad)))

        dataset = Dataset(data_type='dev', label_type=label_type,
                          batch_size=64, eos_index=1,
                          sort_utt=sort_utt, sorta_grad=sorta_grad,
                          progressbar=True, num_gpu=num_gpu)

        tf.reset_default_graph()
        with tf.Session().as_default() as sess:
            print('=> Loading mini-batch...')
            if label_type == 'character_capital_divide':
                map_file_path = '../../metrics/mapping_files/attention/character_to_num_capital.txt'
            else:
                map_file_path = '../../metrics/mapping_files/attention/' + label_type + '_to_num.txt'
            if label_type in ['character', 'character_capital_divide']:
                map_fn = num2char
            else:
                map_fn = num2phone

            for data, next_epoch_flag in dataset(session=sess):
                inputs, labels, inputs_seq_len, labels_seq_len, _ = data
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

                if next_epoch_flag:
                    break


if __name__ == '__main__':
    unittest.main()
