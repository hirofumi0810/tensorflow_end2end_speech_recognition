#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Load dataset for the multitask CTC model.
   You can use the multi-GPU version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import basename
import random
import numpy as np

from experiments.utils.data.inputs.frame_stacking import stack_frame


class DatasetBase(object):

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, batch_size=None):
        return self.__next_mini_batch(batch_size)

    def reset(self):
        """Reset data counter. This is useful when you'd like to evaluate
        overall data during training.
        """
        self.rest = set(range(0, self.data_num, 1))

    def __next_mini_batch(self, batch_size=None):
        """Generate each mini-batch.
        Args:
            batch_size: int, the size of mini-batch
        Returns:
            A tuple of `(inputs, labels, inputs_seq_len, labels_seq_len, input_names)`
                inputs: list of input data of size `[num_gpu, B, T, input_dim]`
                labels_main: list of target labels in the main task, of size `[num_gpu, B, T]`
                labels_sub: list of target labels in the sub task, of size `[num_gpu, B, T]`
                inputs_seq_len: list of length of inputs of size `[num_gpu, B]`
                input_names: list of file name of input data of size `[num_gpu, B]`
            next_epoch_flag: If true, one epoch is finished
        """
        if batch_size is None:
            batch_size = self.batch_size

        next_epoch_flag = False
        padded_value = -1

        while True:
            if next_epoch_flag:
                next_epoch_flag = False

            # Sort all uttrances
            if self.sort_utt:
                if len(self.rest) > batch_size:
                    data_indices = list(self.rest)[:batch_size]
                    self.rest -= set(data_indices)
                else:
                    # Last mini-batch
                    data_indices = list(self.rest)
                    self.rest = set(range(0, self.data_num, 1))
                    next_epoch_flag = True
                    if self.is_training:
                        print('---Next epoch---')
                    self.epoch += 1
                    if self.epoch == self.sort_stop_epoch:
                        self.sort_utt = False

                # Shuffle selected mini-batch
                random.shuffle(data_indices)

            else:
                if len(self.rest) > batch_size:
                    # Randomly sample mini-batch
                    data_indices = random.sample(list(self.rest), batch_size)
                    self.rest -= set(data_indices)
                else:
                    # Last mini-batch
                    data_indices = list(self.rest)
                    self.rest = set(range(0, self.data_num, 1))
                    next_epoch_flag = True
                    if self.is_training:
                        print('---Next epoch---')

                    # Shuffle selected mini-batch
                    random.shuffle(data_indices)

            # Load dataset in mini-batch
            input_list = np.array(list(
                map(lambda path: np.load(path),
                    np.take(self.input_paths, data_indices, axis=0))))
            label_main_list = np.array(list(
                map(lambda path: np.load(path),
                    np.take(self.label_main_paths, data_indices,
                            axis=0))))
            label_sub_list = np.array(list(
                map(lambda path: np.load(path),
                    np.take(self.label_sub_paths, data_indices,
                            axis=0))))
            input_names = list(
                map(lambda path: basename(path).split('.')[0],
                    np.take(self.input_paths, data_indices, axis=0)))

            if self.input_size is None:
                self.input_size = input_list[0].shape[1]
                if self.num_stack is not None and self.num_skip is not None:
                    self.input_size *= self.num_stack

            # Frame stacking
            if self.num_stack is not None and self.num_skip is not None:
                input_list = stack_frame(
                    input_list,
                    self.input_paths[data_indices],
                    self.frame_num_dict,
                    self.num_stack,
                    self.num_skip,
                    progressbar=False)

            # Compute max frame num in mini-batch
            max_frame_num = max(map(lambda x: x.shape[0], input_list))

            # Compute max target label length in mini-batch
            max_seq_len_main = max(map(len, label_main_list))
            max_seq_len_sub = max(map(len, label_sub_list))

            # Initialization
            inputs = np.zeros(
                (len(data_indices), max_frame_num, self.input_size),
                dtype=np.float32)
            if not self.is_test:
                labels_main = np.array(
                    [[padded_value] * max_seq_len_main]
                    * len(data_indices), dtype=np.int32)
                labels_sub = np.array(
                    [[padded_value] * max_seq_len_sub]
                    * len(data_indices), dtype=np.int32)
            else:
                labels_main = [None] * len(data_indices)
                labels_sub = [None] * len(data_indices)
            inputs_seq_len = np.empty((len(data_indices),), dtype=np.int32)

            # Set values of each data in mini-batch
            for i_batch in range(len(data_indices)):
                data_i = input_list[i_batch]
                frame_num = data_i.shape[0]
                inputs[i_batch, :frame_num, :] = data_i
                if not self.is_test:
                    labels_main[i_batch, :len(
                        label_main_list[i_batch])] = label_main_list[i_batch]
                    labels_sub[i_batch, :len(
                        label_sub_list[i_batch])] = label_sub_list[i_batch]
                else:
                    labels_main[i_batch] = label_main_list[i_batch]
                    labels_sub[i_batch] = label_sub_list[i_batch]
                inputs_seq_len[i_batch] = frame_num

            ###############
            # Multi-GPUs
            ###############
            if self.num_gpu > 1:
                # Now we split the mini-batch data by num_gpu
                inputs = np.array_split(inputs, self.num_gpu, axis=0)
                labels_main = np.array_split(labels_main, self.num_gpu, axis=0)
                labels_sub = np.array_split(labels_sub, self.num_gpu, axis=0)
                inputs_seq_len = np.array_split(
                    inputs_seq_len, self.num_gpu, axis=0)
                input_names = np.array_split(input_names, self.num_gpu, axis=0)
            else:
                inputs = inputs[np.newaxis, :, :, :]
                labels_main = labels_main[np.newaxis, :, :]
                labels_sub = labels_sub[np.newaxis, :, :]
                inputs_seq_len = inputs_seq_len[np.newaxis, :]
                input_names = np.array(input_names)[np.newaxis, :]

            yield (inputs, labels_main, labels_sub, inputs_seq_len,
                   input_names), next_epoch_flag
