#! /usr/bin/env python
# -*- coding: utf-8 -*-

""""Plot similarity between adjacent speech frames."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import audioread
import seaborn as sns
plt.style.use('ggplot')
sns.set_style("white")

sys.path.append(os.path.pardir)
from feature_extraction.read_dataset_ctc import DataSet

blue = '#4682B4'


def plot_similarity(sim_list, wav_index, data_type):
    """
    Args:
        sim_list:
        wav_index:
        data_type: train or dev or test
    """
    sim_list = np.array(sim_list)
    times = np.arange(len(sim_list)) * 0.01

    # read wav file
    if data_type == 'train':
        TIMIT_PATH = '/n/sd8/inaguma/corpus/timit/original/train/'
    elif data_type == 'dev':
        return 0
    elif data_type == 'test':
        TIMIT_PATH = '/n/sd8/inaguma/corpus/timit/original/test/'

    speaker_name, file_name = wav_index.split('.')[0].split('_')
    region_paths = [os.path.join(TIMIT_PATH, region_name) for region_name in os.listdir(TIMIT_PATH)]
    for region_path in region_paths:
        speaker_paths = [os.path.join(region_path, speaker_name)
                         for speaker_name in os.listdir(region_path)]
        for speaker_path in speaker_paths:
            if speaker_path.split('/')[-1] == speaker_name:
                file_paths = [os.path.join(speaker_path, file_name)
                              for file_name in os.listdir(speaker_path)]
                for file_path in file_paths:
                    if os.path.basename(file_path).split('.')[0] == file_name:
                        if os.path.basename(file_path).split('.')[-1] == 'wav':
                            wav_path = file_path

    with audioread.audio_open(wav_path) as f:
        # print("ch: %d, fs: %d, duration [s]: %.1f" % (f.channels, f.samplerate, f.duration))
        channel = f.channels
        sampling_rate = f.samplerate
        duration = f.duration
        wav_barray = bytearray()
        for buf in f:
            wav_barray.extend(buf)

        # always read as 16bit
        wav_array = np.frombuffer(wav_barray, dtype=np.int16)
        # convert from short to float
        wav_float = pcm2float(wav_array)

    plt.clf()
    plt.figure(figsize=(10, 4))
    ####################
    # wavform
    ####################
    plt.subplot(211)
    # plt.title(wav_index)
    plt.tick_params(labelleft='off')
    sampling_interval = 1.0 / sampling_rate
    times_wav = np.arange(len(wav_float)) * sampling_interval
    plt.plot(times_wav, wav_float, color='grey')
    plt.ylabel('Amplitude', fontsize=12)
    plt.xlim([0, times_wav[-1]])
    plt.xticks(list(range(0, int(len(sim_list) / 100) + 1, 1)))
    plt.yticks(list(range(0, 2, 1)))

    ####################
    # similarity
    ####################
    plt.subplot(212)
    plt.plot(times, sim_list, color=blue, linewidth=2)
    plt.xlabel('Time[sec]', fontsize=12)
    plt.ylabel('Similarity', fontsize=12)
    plt.xlim([0, duration])
    plt.ylim([0, 1])
    plt.xticks(list(range(0, int(len(sim_list) / 100) + 1, 1)))
    plt.yticks(list(range(0, 2, 1)))
    # plt.legend(loc="upper right", fontsize=12)
    plt.show()


def pcm2float(short_ndary):
    """Convert from short to float."""
    float_ndary = np.array(short_ndary, dtype=np.float64)
    return float_ndary
    # return np.where(float_ndary > 0.0, float_ndary / 32767.0, float_ndary / 32768.0)


def compute_similarity(x, y, sim_type):
    """Compute similarity between two features.
    Args:
        x: first features
        y: second features
        sim_type: cosine or kmeans
    Returns:
        similarity: A float value.
    """
    if sim_type == 'cosine':
        return cosine_sim(x, y)


def cosine_sim(x, y):
    """Compute cosine similarity.
    Args:
        x: ndarray
        y: ndarray
    Returns:
        cos_sim: cosine similarity
    """
    xy_dot = np.dot(x, y)
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    cos_sim = xy_dot / (x_norm * y_norm)

    return cos_sim


def main():
    # read speech frames
    print('Loading dataset...')
    dataset = DataSet(data_type='test', label_type='phone39', is_sorted=True)

    batch_size = 1
    num_examples = dataset.data_num
    if batch_size == 1 or batch_size % 2 == 0:
        iteration = int(num_examples / batch_size)
    else:
        iteration = int(num_examples / batch_size) + 1

    for i_step in range(iteration):
        inputs, labels, seq_len, wav_indices = dataset.next_batch(batch_size=batch_size)

        for i_batch in range(inputs.shape[0]):
            sim_list = []
            inputs_each = inputs[i_batch]

            for i_frame in range(0, inputs_each.shape[0] - 1, 1):
                # compute similarity
                sim = compute_similarity(inputs_each[i_frame],
                                         inputs_each[i_frame + 1],
                                         sim_type='cosine')
                sim_list.append(sim)

            plot_similarity(sim_list, wav_index=wav_indices[i_batch], data_type=dataset.data_type)


if __name__ == '__main__':
    main()
