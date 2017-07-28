#! /usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def num2word(num_list, map_file_path, padded_value=-1):
    """Convert from number to word.
    Args:
        num_list: np.ndarray, list of word indices. batch_size == 1 is
            expected.
        map_file_path: path to the mapping file
        padded_value: int, the value used for padding
    Returns:
        word_list: list of words
    """
    # Read mapping file
    map_dict = {}
    with open(map_file_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            map_dict[int(line[1])] = line[0]

    # Remove padded values
    assert type(num_list) == np.ndarray, 'num_list should be np.ndarray.'
    num_list = np.delete(num_list, np.where(num_list == -1), axis=0)

    # Convert from indices to the corresponding words
    word_list = list(map(lambda x: map_dict[x], num_list))

    return word_list
