#! /usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from tqdm import tqdm


def stack_frame(input_list, input_paths, frame_num_dict, num_stack, num_skip, is_progressbar=False):
    """Stack & skip some frames. This implementation is based on
       https://arxiv.org/abs/1507.06947.
           Sak, Haşim, et al. "Fast and accurate recurrent neural network acoustic models for speech recognition."
           arXiv preprint arXiv:1507.06947 (2015).
    Args:
        input_list: list of input data
        input_paths: list of paths to input data
        frame_num_dict:
            key => utterance index
            value => the number of frames
        num_stack: int, the number of frames to stack
        num_skip: int, the number of frames to skip
        is_progressbar: if True, visualize progressbar
    Returns:
        stacked_input_list: list of frame-stacked inputs
    """
    if num_stack < num_skip:
        raise ValueError('Error: skip must be less than stack.')

    input_size = input_list[0].shape[1]
    utt_num = len(input_paths)

    # setting for progressbar
    iterator = tqdm(range(utt_num)) if is_progressbar else range(utt_num)

    stacked_input_list = []
    for i_utt in iterator:
        # per utterance
        input_name = input_paths[i_utt].split('/')[-1].split('.')[0]
        frame_num = frame_num_dict[input_name]
        frame_num_decimated = frame_num / num_skip
        if frame_num_decimated != int(frame_num_decimated):
            frame_num_decimated += 1
        frame_num_decimated = int(frame_num_decimated)

        stacked_frames = np.zeros(
            (frame_num_decimated, input_size * num_stack))
        stack_count = 0  # counter for stacked_frames
        stack = []
        for i_frame, frame in enumerate(input_list[i_utt]):
            #####################
            # final frame
            #####################
            if i_frame == len(input_list[i_utt]) - 1:
                # stack the final frame
                stack.append(frame)

                while stack_count != int(frame_num_decimated):
                    # concatenate stacked frames
                    for i_stack in range(len(stack)):
                        stacked_frames[stack_count][input_size *
                                                    i_stack:input_size * (i_stack + 1)] = stack[i_stack]
                    stack_count += 1

                    # delete some frames to skip
                    for _ in range(num_skip):
                        if len(stack) != 0:
                            stack.pop(0)

            ########################
            # first & middle frames
            ########################
            elif len(stack) < num_stack:
                # stack some frames until stack is filled
                stack.append(frame)

                if len(stack) == num_stack:
                    # concatenate stacked frames
                    for i_stack in range(num_stack):
                        stacked_frames[stack_count][input_size *
                                                    i_stack:input_size * (i_stack + 1)] = stack[i_stack]
                    stack_count += 1

                    # delete some frames to skip
                    for _ in range(num_skip):
                        stack.pop(0)

        stacked_input_list.append(stacked_frames)

    return stacked_input_list
