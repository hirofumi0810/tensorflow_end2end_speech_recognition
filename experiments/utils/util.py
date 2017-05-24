#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os


def mkdir(path_to_dir):
    """Make a new directory if the directory does not exist.
    Args:
        path_to_dir: path to a directory
    Returns:
        path: path to the new directory
    """
    if not os.path.isdir(path_to_dir):
        os.mkdir(path_to_dir)
    return path_to_dir


def join(path_to_dir, dir_name):
    """concat 2 paths and make a new direcory if the direcory does not exist.
    Args:
        path_to_dir: path to a diretcory
        dir_name: a direcory name
    Returns:
        path to the new directory
    """
    return mkdir(os.path.join(path_to_dir, dir_name))
