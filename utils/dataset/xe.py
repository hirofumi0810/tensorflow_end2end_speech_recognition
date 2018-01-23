#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class for loading dataset for the frame-wise model.
   In this class, all data will be loaded at each step.
   You can use the multi-GPU version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np

from utils.dataset.base import Base


class DatasetBase(Base):

    def __init__(self, *args, **kwargs):
        super(DatasetBase, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        input_i = np.array(self.input_paths[index])
        label_i = np.array(self.label_paths[index])
        return (input_i, label_i)

    def __len__(self):
        if self.data_type == 'train':
            return 18088388
        elif self.data_type == 'dev_clean':
            return 968057
        elif self.data_type == 'dev_other':
            return 919980

    def __next__(self, batch_size=None):
        """Generate each mini-batch.
        Args:
            batch_size (int, optional): the size of mini-batch
        Returns:
            A tuple of `(inputs, labels, inputs_seq_len, labels_seq_len, input_names)`
                inputs: list of input data of size
                    `[num_gpu, B, input_size]`
                labels: list of target labels of size
                    `[num_gpu, B, num_classes]`
                input_names: list of file name of input data of size
                    `[num_gpu, B]`
            is_new_epoch (bool): If true, 1 epoch is finished
        """
        if self.max_epoch is not None and self.epoch >= self.max_epoch:
            raise StopIteration
        # NOTE: max_epoch = None means infinite loop

        if batch_size is None:
            batch_size = self.batch_size

        # reset
        if self.is_new_epoch:
            self.is_new_epoch = False

        # Load the first block at each epoch
        if self.iteration == 0 or self.is_new_epoch:
            # Randomly sample block
            block_index = random.sample(list(self.rest_block), 1)
            self.rest_block -= set(block_index)

            # Load block
            self.inputs_block = np.array(list(
                map(lambda path: np.load(path),
                    self.input_paths[block_index])))
            # NOTE: `[1, num_frames_per_block, input_dim]`
            self.inputs_block = self.inputs_block.reshape(
                -1, self.inputs_block.shape[-1])

            self.labels_block = np.array(list(
                map(lambda path: np.load(path),
                    self.label_paths[block_index])))
            # NOTE: `[1, num_frames_per_block, num_classes]`
            self.labels_block = self.labels_block.reshape(
                -1, self.labels_block.shape[-1])

            self.rest_frames = set(range(0, len(self.inputs_block), 1))

        # Load block if needed
        if len(self.rest_frames) < batch_size and len(self.rest_block) != 0:
            # Randomly sample block
            if len(self.rest_block) > 1:
                block_index = random.sample(list(self.rest_block), 1)
            else:
                # Last block in each epoch
                block_index = list(self.rest_block)
            self.rest_block -= set(block_index)

            # tmp
            rest_inputs_pre_block = self.inputs_block[list(self.rest_frames)]
            rest_labels_pre_block = self.labels_block[list(self.rest_frames)]

            self.inputs_block = np.array(list(
                map(lambda path: np.load(path),
                    self.input_paths[block_index]))).reshape(-1, self.inputs_block.shape[-1])
            self.labels_block = np.array(list(
                map(lambda path: np.load(path),
                    self.label_paths[block_index]))).reshape(-1, self.labels_block.shape[-1])

            # Concatenate
            self.inputs_block = np.concatenate(
                (rest_inputs_pre_block, self.inputs_block), axis=0)
            self.labels_block = np.concatenate(
                (rest_labels_pre_block, self.labels_block), axis=0)

            self.rest_frames = set(range(0, len(self.inputs_block), 1))

        # Randomly sample frames
        if len(self.rest_frames) > batch_size:
            frame_indices = random.sample(
                list(self.rest_frames), batch_size)
        else:
            # Last mini-batch in each block
            frame_indices = list(self.rest_frames)

            # Shuffle selected mini-batch
            random.shuffle(frame_indices)
        self.rest_frames -= set(frame_indices)

        if len(self.rest_block) == 0 and len(self.rest_frames) == 0:
            self.reset()
            self.is_new_epoch = True
            self.epoch += 1
            self.rest_block = set(range(0, len(self.input_paths), 1))

        # Set values of each data in mini-batch
        inputs = self.inputs_block[frame_indices]
        labels = self.labels_block[frame_indices]

        ###############
        # Multi-GPUs
        ###############
        if self.num_gpu > 1:
            # Now we split the mini-batch data by num_gpu
            inputs = np.array_split(inputs, self.num_gpu, axis=0)
            labels = np.array_split(labels, self.num_gpu, axis=0)
        else:
            inputs = inputs[np.newaxis, :, :]
            labels = labels[np.newaxis, :, :]

        self.iteration += len(frame_indices)

        return (inputs, labels), self.is_new_epoch
