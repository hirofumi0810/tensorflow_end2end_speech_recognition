#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class for loading dataset for the Attention model.
   You can use the multi-GPU version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import basename
import random
import numpy as np
import tensorflow as tf


class DatasetBase(object):

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, batch_size=None, session=None):
        return self.__next_mini_batch(batch_size, session)

    def __next_mini_batch(self, batch_size=None, session=None):
        """Generate each mini-batch.
        Args:
            batch_size: int, the size of mini-batch
            session: set when using multiple GPUs
        Returns:
            A tuple of `(inputs, labels, inputs_seq_len, labels_seq_len, input_names)`
                inputs: list of input data of size `[num_gpu, B, T, input_dim]`
                labels: list of target labels of size `[num_gpu, B, T]`
                inputs_seq_len: list of length of inputs of size `[num_gpu, B]`
                labels_seq_len: list of length of target labels of size `[num_gpu, B]`
                input_names: list of file name of input data of size `[num_gpu, B]`
            next_epoch_flag: If true, one epoch is finished
        """
        if session is None and self.num_gpu != 1:
            raise ValueError('Set session when using multiple GPUs.')

        if batch_size is None:
            batch_size = self.batch_size

        next_epoch_flag = False

        while True:
            if next_epoch_flag:
                next_epoch_flag = False

            # Sort all uttrances
            if self.sort_utt or self.sorta_grad:
                if len(self.rest) > batch_size:
                    data_indices = list(self.rest)[:batch_size]
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
                if len(self.rest) > batch_size:
                    # Randomly sample mini-batch
                    data_indices = random.sample(
                        list(self.rest), batch_size)
                    self.rest -= set(data_indices)
                else:
                    data_indices = list(self.rest)
                    self.rest = set(range(0, self.data_num, 1))
                    next_epoch_flag = True
                    if self.data_type == 'train':
                        print('---Next epoch---')

                    # Shuffle selected mini-batch
                    random.shuffle(data_indices)

            # Load dataset in mini-batch
            input_list = np.array(list(
                map(lambda path: np.load(path),
                    np.take(self.input_paths, data_indices, axis=0))))
            label_list = np.array(list(
                map(lambda path: np.load(path),
                    np.take(self.label_paths, data_indices, axis=0))))
            input_names = list(
                map(lambda path: basename(path).split('.')[0],
                    np.take(self.input_paths, data_indices, axis=0)))
            if self.input_size is None:
                self.input_size = input_list[0].shape[1]
                if self.num_stack is not None and self.num_skip is not None:
                    self.input_size *= self.num_stack

            # Compute max frame num in mini-batch
            max_frame_num = max(map(lambda x: x.shape[0], input_list))

            # Compute max target label length in mini-batch
            max_seq_len = max(map(len, label_list))

            # Initialization
            inputs = np.zeros(
                (len(data_indices), max_frame_num, self.input_size),
                dtype=np.float32)
            # Padding with <EOS>
            if not self.is_test:
                labels = np.array([[self.eos_index] * max_seq_len]
                                  * len(data_indices), dtype=np.int32)
            else:
                labels = [None] * len(data_indices)
            inputs_seq_len = np.empty(
                (len(data_indices),), dtype=np.int32)
            labels_seq_len = np.zeros(
                (len(data_indices),), dtype=np.int32)

            # Set values of each data in mini-batch
            for i_batch in range(len(data_indices)):
                data_i = input_list[i_batch]
                frame_num = data_i.shape[0]
                inputs[i_batch, : frame_num, :] = data_i
                if not self.is_test:
                    labels[i_batch, :len(label_list[i_batch])
                           ] = label_list[i_batch]
                else:
                    labels[i_batch] = label_list[i_batch]
                inputs_seq_len[i_batch] = frame_num
                labels_seq_len[i_batch] = len(label_list[i_batch])

            ###############
            # Multi-GPUs
            ###############
            if self.num_gpu > 1:
                divide_num = self.num_gpu
                if next_epoch_flag:
                    for i in range(self.num_gpu, 0, -1):
                        if len(data_indices) % i == 0:
                            divide_num = i
                            break

                # Now we split the mini-batch data by num_gpu
                inputs = tf.split(inputs, divide_num, axis=0)
                labels = tf.split(labels, divide_num, axis=0)
                inputs_seq_len = tf.split(
                    inputs_seq_len, divide_num, axis=0)
                labels_seq_len = tf.split(
                    labels_seq_len, divide_num, axis=0)
                input_names = tf.split(input_names, divide_num, axis=0)

                # Convert from SparseTensor to numpy.ndarray
                inputs = list(map(session.run, inputs))
                labels = list(map(session.run, labels))
                inputs_seq_len = list(map(session.run, inputs_seq_len))
                labels_seq_len = list(map(session.run, labels_seq_len))
                input_names = np.array(list(map(session.run, input_names)))
            else:
                inputs = inputs[np.newaxis, :, :, :]
                labels = labels[np.newaxis, :, :]
                inputs_seq_len = inputs_seq_len[np.newaxis, :]
                labels_seq_len = labels_seq_len[np.newaxis, :]
                input_names = np.array(input_names)[np.newaxis, :]

            yield (inputs, labels, inputs_seq_len, labels_seq_len,
                   input_names), next_epoch_flag
