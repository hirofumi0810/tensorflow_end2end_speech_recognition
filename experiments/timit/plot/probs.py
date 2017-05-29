#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot posteriors of CTC outputs (TIMIT corpus)."""

import os
import numpy as np
# import scipy.io.wavfile
import matplotlib.pyplot as plt
import audioread
import seaborn as sns
plt.style.use('ggplot')
sns.set_style("white")

blue = '#4682B4'
orange = '#D2691E'
green = '#006400'


def plot_probs_ctc_phone(probs, save_path, wav_index, data_type, label_type):
    """Plot posteriors of phones.
    Args:
        probs:
        save_path:
        wav_index: int
        data_type: train or dev or test
        label_type: phone39 or phone48 or phone61 or character
    """
    # read wav file
    if data_type == 'train':
        TIMIT_PATH = '/n/sd8/inaguma/corpus/timit/original/train/'
    elif data_type == 'dev':
        return 0
    elif data_type == 'test':
        TIMIT_PATH = '/n/sd8/inaguma/corpus/timit/original/test/'

    speaker_name, file_name = wav_index.split('.')[0].split('_')
    region_paths = [os.path.join(TIMIT_PATH, region_name)
                    for region_name in os.listdir(TIMIT_PATH)]
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

    times_probs = np.arange(len(probs)) * 0.01
    plt.clf()
    plt.figure(figsize=(10, 4))

    ####################
    # waveform
    ####################
    plt.subplot(211)
    plt.title(wav_index)
    plt.tick_params(labelleft='off')
    sampling_interval = 1.0 / sampling_rate
    # wav_float = wav_float / 32768.0
    times_wav = np.arange(len(wav_float)) * sampling_interval
    plt.plot(times_wav, wav_float, color='grey')
    plt.ylabel('Amplitude', fontsize=12)
    plt.xlim([0, times_wav[-1]])
    plt.xticks(list(range(0, int(len(probs) / 100) + 1, 1)))
    plt.yticks(list(range(0, 2, 1)))

    ####################
    # phones
    ####################
    plt.subplot(212)
    plt.plot(times_probs, probs[:, 0],
             label='silence', color='black', linewidth=2)
    if label_type == 'phone39':
        blank_index = 39
    elif label_type == 'phone48':
        blank_index = 48
    elif label_type == 'phone61':
        blank_index = 61
    for i in range(1, blank_index, 1):
        plt.plot(times_probs, probs[:, i])
    plt.plot(times_probs, probs[:, blank_index],
             ':', label='blank', color='grey')
    plt.xlabel('Time[sec]', fontsize=12)
    plt.ylabel('Phones', fontsize=12)
    plt.xlim([0, duration])
    plt.ylim([0.05, 1.05])
    plt.xticks(list(range(0, int(len(probs) / 100) + 1, 1)))
    plt.yticks(list(range(0, 2, 1)))
    plt.legend(loc="upper right", fontsize=12)

    # save_path = os.path.join(save_path, wav_index + '.png')
    # plt.savefig(save_path, dvi=500)
    plt.show()


def pcm2float(short_ndary):
    """Convert from short to float."""
    float_ndary = np.array(short_ndary, dtype=np.float64)
    return float_ndary
    # return np.where(float_ndary > 0.0, float_ndary / 32767.0, float_ndary /
    # 32768.0)
