#! /usr/bin/env python
# -*- coding: utf-8 -*-


def char2num(str_char, map_file_path):
    """Convert from character to number.
    Args:
        str_char: string of characters
        map_file_path: path to the mapping file
    Returns:
        char_list: list of character indices
    """
    char_list = list(str_char)

    # read mapping file
    map_dict = {}
    with open(map_file_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            map_dict[line[0]] = int(line[1])

    # convert from character to number
    for i in range(len(char_list)):
        char_list[i] = map_dict[char_list[i]]

    return char_list


def num2char(num_list, map_file_path):
    """Convert from number to character.
    Args:
        num_list: list of character indices
        map_file_path: path to the mapping file
    Returns:
        str_char: string of characters
    """
    # read mapping file
    map_dict = {}
    with open(map_file_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            map_dict[int(line[1])] = line[0]

    # convert from indices to the corresponding characters
    char_list = []
    for i in range(len(num_list)):
        char_list.append(map_dict[num_list[i]])

    str_char = ''.join(char_list)
    return str_char
