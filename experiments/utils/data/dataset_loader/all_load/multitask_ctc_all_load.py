#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class for loading dataset for the multi-task CTC model.
   In this class, all data will be loaded at once.
   You can use only the single GPU version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import basename
import random
import numpy as np


class DatasetBase(object):

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, batch_size=None):
        return self.__next_mini_batch(batch_size)

    def __next_mini_batch(self, _batch_size):
        """Generate each mini-batch.
        Args:
            _batch_size: int, the size of mini-batch
            session: set when using multiple GPUs
        Returns:
            A tuple of `(inputs, labels, inputs_seq_len, labels_seq_len, input_names)`
                inputs: list of input data of size `[B, T, input_dim]`
                labels_main: list of target labels in the main task, of size `[B, T]`
                labels_sub: list of target labels in the sub task, of size `[B, T]`
                inputs_seq_len: list of length of inputs of size `[B]`
                input_names: list of file name of input data of size `[B]`
            next_epoch_flag: If true, one epoch is finished
        """
        if _batch_size is None:
            _batch_size = self.batch_size

        next_epoch_flag = False
        padded_value = -1

        while True:
            if next_epoch_flag:
                next_epoch_flag = False

            # Sort all uttrances in each epoch
            if self.sort_utt or self.sorta_grad:
                if len(self.rest) > _batch_size:
                    data_indices = list(self.rest)[:_batch_size]
                    self.rest -= set(data_indices)
                else:
                    data_indices = list(self.rest)
                    self.rest = set(range(0, self.data_num, 1))
                    next_epoch_flag = True
                    if self.data_type == 'train':
                        print('---Next epoch---')
                    if self.sorta_grad:
                        self.sorta_grad = False

                # Shuffle selected mini-batch
                if not self.sorta_grad:
                    random.shuffle(data_indices)

            else:
                if len(self.rest) > _batch_size:
                    # Randomly sample mini-batch
                    data_indices = random.sample(
                        list(self.rest), _batch_size)
                    self.rest -= set(data_indices)
                else:
                    data_indices = list(self.rest)
                    self.rest = set(range(0, self.data_num, 1))
                    next_epoch_flag = True
                    if self.data_type == 'train':
                        print('---Next epoch---')

                    # Shuffle selected mini-batch
                    random.shuffle(data_indices)

            # Compute max frame num in mini-batch
            max_frame_num = max(map(lambda x: x.shape[0],
                                    self.input_list[data_indices]))

            # Compute max target label length in mini-batch
            max_seq_len_main = max(map(len,
                                       self.label_main_list[data_indices]))
            max_seq_len_sub = max(map(len,
                                      self.label_sub_list[data_indices]))

            # Initialization
            inputs = np.zeros(
                (len(data_indices), max_frame_num, self.input_size),
                dtype=np.int32)
            labels_main = np.array([[padded_value] * max_seq_len_main]
                                   * len(data_indices), dtype=np.int32)
            labels_sub = np.array([[padded_value] * max_seq_len_sub]
                                  * len(data_indices), dtype=np.int32)
            inputs_seq_len = np.empty((len(data_indices),), dtype=np.int32)
            input_names = np.array(list(
                map(lambda path: basename(path).split('.')[0],
                    np.take(self.input_paths, data_indices, axis=0))))

            # Set values of each data in mini-batch
            for i_batch, x in enumerate(data_indices):
                data_i = self.input_list[x]
                frame_num = data_i.shape[0]
                inputs[i_batch, :frame_num, :] = data_i
                labels_main[i_batch, :len(
                    self.label_main_list[x])] = self.label_main_list[x]
                labels_sub[i_batch, :len(
                    self.label_sub_list[x])] = self.label_sub_list[x]
                inputs_seq_len[i_batch] = frame_num

            yield (inputs, labels_main, labels_sub, inputs_seq_len,
                   input_names), next_epoch_flag
