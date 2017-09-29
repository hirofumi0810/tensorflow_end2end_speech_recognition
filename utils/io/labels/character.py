#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class Char2idx(object):
    """Convert from character to index.
    Args:
        map_file_path (string): path to the mapping file
    """

    def __init__(self, str_char, map_file_path):
        # Read the mapping file
        self.map_dict = {}
        with open(map_file_path, 'r') as f:
            for line in f:
                line = line.strip().split()
                self.map_dict[line[0]] = int(line[1])

    def __call__(self, str_char):
        """
        Args:
            str_char (string): a sequence of characters
        Returns:
            index_list (list): character indices
        """
        char_list = list(str_char)

        # Convert from character to index
        index_list = list(map(lambda x: self.map_dict[x], char_list))

        return np.array(index_list)


class Kana2idx(object):
    """Convert from kana character to index.
    Args:
        map_file_path (string): path to the mapping file
    """

    def __init__(self, map_file_path):
        # Read the mapping file
        self.map_dict = {}
        with open(map_file_path, 'r') as f:
            for line in f:
                line = line.strip().split()
                self.map_dict[line[0]] = int(line[1])

    def __call__(self, str_char):
        """
        Args:
            str_char (string): a sequence of kana characters
        Returns:
            index_list (list): kana character indices
        """
        kana_list = list(str_char)
        index_list = []

        for i in range(len(kana_list)):
            # Check whether next kana character is a double consonant
            if i != len(kana_list) - 1:
                if kana_list[i] + kana_list[i + 1] in self.map_dict.keys():
                    index_list.append(
                        int(self.map_dict[kana_list[i] + kana_list[i + 1]]))
                    i += 1
                elif kana_list[i] in self.map_dict.keys():
                    index_list.append(int(self.map_dict[kana_list[i]]))
                else:
                    raise ValueError(
                        'There are no kana character such as %s' % kana_list[i])
            else:
                if kana_list[i] in self.map_dict.keys():
                    index_list.append(int(self.map_dict[kana_list[i]]))
                else:
                    raise ValueError(
                        'There are no kana character such as %s' % kana_list[i])

        return np.array(index_list)


class Idx2char(object):
    """Convert from index to character.
    Args:
        map_file_path (string): path to the mapping file
        capital_divide (bool, optional): set True when using capital-divided
            character sequences
        space_mark (string): the space mark to divide a sequence into words
    """

    def __init__(self, map_file_path, capital_divide=False, space_mark=' '):
        self.capital_divide = capital_divide
        self.space_mark = space_mark

        # Read the mapping file
        self.map_dict = {}
        with open(map_file_path, 'r') as f:
            for line in f:
                line = line.strip().split()
                self.map_dict[int(line[1])] = line[0]

    def __call__(self, index_list, padded_value=-1):
        """
        Args:
            index_list (np.ndarray): list of character indices.
                Batch size 1 is expected.
            padded_value (int): the value used for padding
        Returns:
            str_char (string): a sequence of characters
        """
        # Remove padded values
        assert type(index_list) == np.ndarray, 'index_list should be np.ndarray.'
        index_list = np.delete(index_list, np.where(
            index_list == padded_value), axis=0)

        # Convert from indices to the corresponding characters
        char_list = list(map(lambda x: self.map_dict[x], index_list))

        if self.capital_divide:
            char_list_tmp = []
            for i in range(len(char_list)):
                if i != 0 and 'A' <= char_list[i] <= 'Z':
                    char_list_tmp += [self.space_mark, char_list[i].lower()]
                else:
                    char_list_tmp += [char_list[i].lower()]
            str_char = ''.join(char_list_tmp)
        else:
            str_char = ''.join(char_list)

        return str_char
        # TODO: change to batch version
