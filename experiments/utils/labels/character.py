#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def char2num(str_char, map_file_path):
    """Convert from character to number.
    Args:
        str_char: string of characters
        map_file_path: path to the mapping file
    Returns:
        char_list: list of character indices
    """
    char_list = list(str_char)

    # Read mapping file
    map_dict = {}
    with open(map_file_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            map_dict[line[0]] = int(line[1])

    # Convert from character to number
    for i in range(len(char_list)):
        char_list[i] = map_dict[char_list[i]]

    return char_list


def num2char(num_list, map_file_path, padded_value=-1):
    """Convert from number to character.
    Args:
        num_list: np.ndarray, list of character indices. batch_size == 1 is
            expected.
        map_file_path: path to the mapping file
        padded_value: int, the value used for padding
    Returns:
        str_char: string of characters
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

    # Convert from indices to the corresponding characters
    char_list = list(map(lambda x: map_dict[x], num_list))

    str_char = ''.join(char_list)
    return str_char
    # TODO: change to batch version
