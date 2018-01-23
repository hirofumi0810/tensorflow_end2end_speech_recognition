#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import unittest

sys.path.append(os.path.abspath('../../../../'))
from experiments.librispeech.data.load_dataset_xe import Dataset
from utils.measure_time_func import measure_time


class TestLoadDatasetCTC(unittest.TestCase):

    def test(self):

        # data_type
        self.check(data_type='train')
        self.check(data_type='dev_clean')
        self.check(data_type='dev_other')

    @measure_time
    def check(self, data_type='dev_clean'):

        print('========================================')
        print('  data_type: %s' % data_type)
        print('========================================')

        dataset = Dataset(
            model_path='/speech7/takashi01_nb/inaguma/models/tensorflow/librispeech/ctc/character/train100h/blstm_ctc_320_5_rmsprop_lr1e-3_drop0.2_stack2_temp1_7/temp1',
            data_type=data_type,
            batch_size=512, max_epoch=4,
            num_gpu=1)

        print('=> Loading mini-batch...')
        for data, is_new_epoch in dataset:
            inputs, labels = data

            print('----- (epoch_detail: %.3f) -----' %
                  (dataset.epoch_detail))
            print(inputs[0].shape)
            print(labels[0].shape)

            # if dataset.epoch_detail >= 0.1:
            #     break


if __name__ == '__main__':
    unittest.main()
