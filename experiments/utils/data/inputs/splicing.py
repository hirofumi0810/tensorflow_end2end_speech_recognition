#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Splice data."""

import numpy as np


def do_splice(input_data, splice=0):
    """Splice input data. This is expected to be used for DNNs or RNNs.
    Args:
        input_data: list of size `[max_time, input_size]' in each timestep
        splice: int, the frame number to splice
            ex.) splice==2
                [t-2, t-1, t, t+1, t+2]
    Returns:
        data_spliced: np.ndarray of size
            `[max_time, input_size * (2 * splice + 1)]`
    """
    assert isinstance(
        input_data, np.ndarray), 'input_data should be np.ndarray.'

    if splice == 0:
        return input_data

    max_time = input_data.shape[0]
    input_size = input_data.shape[1]
    input_data_spliced = np.zeros((max_time, input_size * (2 * splice + 1)))

    for t in range(max_time):
        for i in range(0, 2 * splice + 1, 1):
            #########################
            # padding left frames
            #########################
            if t <= splice - 1 and i < splice - t:
                # copy input_data[0] to left side
                copy_frame = input_data[0]

            #########################
            # padding right frames
            #########################
            elif max_time - splice <= t and t + (i - splice) > max_time - 1:
                # copy input_data[-1] to right side
                copy_frame = input_data[-1]

            #########################
            # middle of frames
            #########################
            else:
                copy_frame = input_data[t + (i - splice)]

            input_data_spliced[t][input_size *
                                  i: input_size * (i + 1)] = copy_frame

    return input_data_spliced


def do_image_splice(input_data, splice=0):
    """Splice input data and return images. This is expected to be used for CNNs.
    Args:
        input_data: list of size `[max_time, input_size]' in each timestep
        splice: int, the frame number to splice
            ex.) splice==2
                [t-2, t-1, t, t+1, t+2]
    Returns:
        data_spliced: np.ndarray of size
            `[max_time, input_size, 2 * splice + 1]`
    """
    assert isinstance(
        input_data, np.ndarray), 'input_data should be np.ndarray.'

    max_time = input_data.shape[0]
    input_size = input_data.shape[1]

    if splice == 0:
        # Reshape to images
        return input_data.reshape((max_time, input_size, 1))

    input_data_spliced = np.zeros((max_time, 2 * splice + 1, input_size))

    for t in range(max_time):
        for i in range(0, 2 * splice + 1, 1):
            #########################
            # padding left frames
            #########################
            if t <= splice - 1 and i < splice - t:
                # copy input_data[0] to left side
                copy_frame = input_data[0]

            #########################
            # padding right frames
            #########################
            elif max_time - splice <= t and t + (i - splice) > max_time - 1:
                # copy input_data[-1] to right side
                copy_frame = input_data[-1]

            #########################
            # middle of frames
            #########################
            else:
                copy_frame = input_data[t + (i - splice)]

            input_data_spliced[t][i] = copy_frame

    return input_data_spliced.transpose(0, 2, 1)


def test():

    sequence = np.zeros((100, 5))
    for i in range(sequence.shape[0]):
        sequence[i][0] = i
    sequence_spliced = do_splice(sequence, splice=5)
    assert sequence_spliced.shape == (100, 5 * (2 * 5 + 1)), "Error"

    # for i in range(sequence_spliced.shape[0]):
    #     print(sequence_spliced[i])

    sequence_spliced = do_image_splice(sequence, splice=5)
    assert sequence_spliced.shape == (100, 5, 2 * 5 + 1), "Error"

    # for i in range(sequence_spliced.shape[0]):
    #     print(sequence_spliced[i].shape)


if __name__ == '__main__':
    test()
