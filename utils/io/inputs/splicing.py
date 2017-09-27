#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Splice data."""

import numpy as np


def do_splice(inputs, splice=1, batch_size=1):
    """Splice input data. This is expected to be used for DNNs or RNNs.
    Args:
        inputs (np.ndarray): list of size `[B, T, input_size]'
        splice (int): frames to splice. Default is 1 frame.
            ex.) splice == 11
                [t-5, ..., t-1, t, t+1, ..., t+5] (total 11 frames)
        batch_size: int,
    Returns:
        data_spliced (np.ndarray): A tensor of size
            `[B, T, input_size * splice]`
    """
    assert isinstance(inputs, np.ndarray), 'inputs should be np.ndarray.'
    # assert len(inputs.shape) == 3, 'inputs must be 3 demension.'

    if splice == 1:
        return inputs

    batch_size, max_time, input_size = inputs.shape
    input_data_spliced = np.zeros((batch_size, max_time, input_size * splice))

    for i_batch in range(batch_size):
        for i_time in range(max_time):
            for i_splice in range(0, splice, 1):
                #########################
                # padding left frames
                #########################
                if i_time <= splice - 1 and i_splice < splice - i_time:
                    # copy the first frame to left side
                    copy_frame = inputs[i_batch][0]

                #########################
                # padding right frames
                #########################
                elif max_time - splice <= i_time and i_time + (i_splice - splice) > max_time - 1:
                    # copy the last frame to right side
                    copy_frame = inputs[i_batch][-1]

                #########################
                # middle of frames
                #########################
                else:
                    copy_frame = inputs[i_batch][i_time + (i_splice - splice)]

                input_data_spliced[i_batch][i_time][input_size *
                                                    i_splice: input_size * (i_splice + 1)] = copy_frame

    return input_data_spliced


def test():
    sequence = np.zeros((3, 100, 5))
    for i_batch in range(sequence.shape[0]):
        for i_frame in range(sequence.shape[1]):
            sequence[i_batch][i_frame][0] = i_frame
    sequence_spliced = do_splice(sequence, splice=11)
    assert sequence_spliced.shape == (3, 100, 5 * 11)

    # for i in range(sequence_spliced.shape[1]):
    #     print(sequence_spliced[0][i])


if __name__ == '__main__':
    test()
