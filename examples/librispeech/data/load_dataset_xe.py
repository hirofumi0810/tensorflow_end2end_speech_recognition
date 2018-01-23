#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Load dataset for the frame-wise model (Librispeech corpus).
   You can use the multi-GPU version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import numpy as np
from glob import glob

from utils.dataset.xe import DatasetBase


class Dataset(DatasetBase):

    def __init__(self, model_path, data_type, batch_size, max_epoch=None,
                 num_gpu=1):
        """A class for loading dataset.
        Args:
            model_path (string): path to the saved model
            data_type (string): character
            batch_size (int): the size of mini-batch
            max_epoch (int, optional): the max epoch. None means infinite loop.
            num_gpu (int, optional): if more than 1, divide batch_size by num_gpu
        """
        super(Dataset, self).__init__()

        self.model_path = model_path
        self.data_type = data_type
        self.batch_size = batch_size * num_gpu
        self.max_epoch = max_epoch
        self.num_gpu = num_gpu

        input_path = join(model_path, data_type, 'inputs')
        label_path = join(model_path, data_type, 'labels')
        # NOTE: block0.npy, block1.npy ...

        # Sort paths to input & label
        input_paths, label_paths = [], []
        for file_name in glob(join(input_path, '*.npy')):
            input_paths.append(file_name)
        for file_name in glob(join(label_path, '*.npy')):
            label_paths.append(file_name)
        self.input_paths = np.array(input_paths)
        self.label_paths = np.array(label_paths)
        # NOTE: Not load dataset yet

        self.rest_block = set(range(0, len(self.input_paths), 1))
        # NOTE: len(self.rest_block) == num_blocks

        self.rest_frames = set([])
