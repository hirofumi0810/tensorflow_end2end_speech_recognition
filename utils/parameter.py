#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def count_total_parameters(variables):
    """
    Args:
        variables (list): tf.trainable_variables()
    Returns:
        parameters_dict (dict):
            key => variable name
            value => the number of parameters
        total_parameters (float): total parameters of the model
    """
    total_parameters = 0
    parameters_dict = {}
    for variable in variables:
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
        parameters_dict[variable.name] = variable_parameters
    return parameters_dict, total_parameters
